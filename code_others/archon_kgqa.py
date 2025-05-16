from archon.completions.utils import custom_generator, custom_vision_generator
from archon.completions import Archon # 'Archon' is the initialization class
from archon.completions.components import Component, Generator #

import os, sys
import argparse
import json
import tqdm
import re
from collections import Counter
import numpy as np
import wandb

from utils import precision_recall_f1, accuracy
from evaluation_2wiki import evaluate, metric_evaluation

api_keys = {
    "OPENAI_API_KEY": [
        "your_api_key"
    ],
    "TOGETHER_API_KEY": [
        "your_api_key"
    ],
    "ANTHROPIC_API_KEY": [
        "your_api_key"
    ],
}
model_size_mapping = {"Qwen/Qwen2.5-7B-Instruct-AWQ":7, "Qwen/Qwen2.5-32B-Instruct-AWQ":32, "Qwen/Qwen2.5-72B-Instruct-AWQ":72,
"meta-llama/Llama-3.2-3B-Instruct": 3, "meta-llama/Llama-3.1-8B-Instruct": 8, "meta-llama/Llama-3.1-70B-Instruct": 70, "google/gemma-2-2b-it": 2, "google/gemma-2-9b-it": 9, "google/gemma-2-27b-it": 27, "microsoft/Phi-3-mini-4k-instruct": 3.8, "microsoft/Phi-3-small-8k-instruct": 7, "microsoft/Phi-3-medium-4k-instruct": 14, "infly/OpenCoder-1.5B-Instruct": 1.5, "infly/OpenCoder-8B-Instruct": 8
}
print_log = False

class Retrieval(Component):

    def __init__(self, config, custom_components=None):
        self.config = config

        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.samples = self.config["samples"]
        self.custom_components = custom_components

        # Generator class helps us handle messaging
        self.model = Generator(self.config, custom_generators=custom_components)
        print(f"Retrieval initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):
        """
        Run the component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon.
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """

        # print('Log, conversation:', conversation)
        # print()

        candidates = prev_state["candidates"]
        retrieval_results = self.retrieve(conversation, candidates)
        state["candidates"].extend(retrieval_results)

        return

    def retrieve(self, conversation: list, candidates: list):

        # conversation is the initial start
        query = conversation[-1]["content"]
        knowledge_triplets = conversation[-1]["knowledge_triplets"]
        knowledge_triplets_str = "<head entity, relation, tail entity>\n"
        conversation[-1]['knowledge_triplets_str'] = knowledge_triplets_str
        for i, triplet in enumerate(knowledge_triplets):
            s = f"[{i+1}] <{triplet[0]}, {triplet[1]}, {triplet[2]}>\n"
            knowledge_triplets_str += s

        prompt = (f"You have been provided with a query and a set of knowledge triplets from a knowledge graph. "
                  f"Your task is to select knowledge triplets that help answer the question. "
                  f"Your response should be a list of selected knowledge triplets in the following format: Selected knowledge triplets: [reference number] [corresponding knowledge triplet]' and no other words."
                  f"###Query: {query}.\n\n"
                  f"###Knowledge triplets: {knowledge_triplets_str}\n"
                  )
        if print_log:
            print('\nLog, Retrieval, prompt:', prompt)
            print('\n', '### Log, retrieve, len(prompt.split())', len(prompt.split()), '\n')

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": prompt}]  # prompt
        )

        generations = []
        output = self.model.generate_from_messages(
            messages,
            temperature=self.temperature,
        )
        if output is not None:
            generations.extend(output)
        if print_log:
            print('\n', '### Log, retrieve, len(output.split())', len(output[0].split()), '\n')

        return generations


class Merge(Component):
    '''
    for Merge, the samples means calculate the metric based on how many candidates
    '''

    def __init__(self, config, custom_components=None):
        self.config = config

        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.samples = self.config["samples"]
        self.custom_components = custom_components


    def run(self, conversation: list, prev_state, state):
        """
        Run the component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon.
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """
        candidates = prev_state["candidates"]
        if print_log: print('Log, candidates to Merge:', candidates)

        query = conversation[-1]["content"]
        knowledge_triplets = conversation[-1]['knowledge_triplets']

        # first time, generate many candidates following Retrieval component and save to retrieval_save_file
        # second time, only use Merge component, load the candidates from retrieval_save_file files.

        ref_ids = []
        lengths = []
        for candidate in candidates:
            numbers = re.findall(r'\[(\d+)\]', candidate)
            ref_ids.extend(numbers)
            lengths.append(len(numbers))
        if len(ref_ids) > 0:
            most_common_length = Counter(lengths).most_common()[0][0]
            selected_ids = [e[0] for e in Counter(ref_ids).most_common()[0:most_common_length]]
        else:
            selected_ids = [1]
        selected_ids = [int(e) for e in selected_ids]
        conversation[-1]["selected_ids"] = selected_ids
        selected_ids_str = ""
        for e in selected_ids:
            if e > len(knowledge_triplets): continue
            selected_ids_str += f"[{str(e)}] " + f"<{knowledge_triplets[e-1][0]}, {knowledge_triplets[e-1][1]}, {knowledge_triplets[e-1][2]}>" + '\n'
        generations = [f"Selected knowledge triplets: {selected_ids_str}"]
        if print_log: print('\nMerge results:', generations)
        state["candidates"].extend(generations)

        return

class CustomGenerator(Component):
    '''
    custom generator, it will receive the selected references and the query.
    '''

    def __init__(self, config, custom_components=None):
        self.config = config

        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.samples = self.config["samples"]
        self.custom_components = custom_components

        # Generator class helps us handle messaging
        self.model = Generator(self.config, custom_generators=custom_components)
        print(f"Retrieval initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):
        """
        Run the component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon.
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """

        candidates = prev_state["candidates"]
        generations = self.generate(conversation, candidates)
        state["candidates"].extend(generations)

        return

    def generate(self, conversation: list, candidates: list):
        '''
        return a list.
        :param conversation:
        :param candidates:
        :return:
        '''
        # conversation is the initial start
        query = conversation[-1]["content"]
        knowledge_triplets = conversation[-1]['knowledge_triplets']
        knowledge_triplets_str = candidates[0]

        selected_ids = conversation[-1]["selected_ids"]  # string list
        ind = 1
        knowledge_triplets_str = ""
        for s_id in selected_ids:
            if s_id > len(knowledge_triplets):
                continue
            knowledge_triplet = knowledge_triplets[s_id-1]
            knowledge_triplets_str += f"[{ind}] " + f"<{knowledge_triplet[0]}, {knowledge_triplet[1]}, {knowledge_triplet[2]}>\n"
            ind += 1

        prompt = (f"You will be provided with a set of knowledge triplets related to the query. "
                  f"Your task is to answer the query using the provided knowledge. If the references are insufficient, supplement your response with your own knowledge. "
                  "Please conclude your response with the sentence: 'Therefore, based on this information, the final answer is '"
                  f"Here is the query and knowledge triplets: \n"
                  f"###Query: {query}\n"
                  f"###Knowledge triplets: {knowledge_triplets_str}\n")
        if print_log:
            print('\nLog, prompt:', prompt)
            print('\n', '### Log, generate, len(prompt.split())', len(prompt.split()), '\n')
        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": prompt}]  # prompt
        )

        generations = []
        # for _ in range(self.samples):
        output = self.model.generate_from_messages(
            messages,
            temperature=self.temperature,
        )
        if output is not None:
            generations.extend(output)
        if print_log:
            print('\n', '### Log, generate, len(output.split())', len(output[0].split()), '\n')

        return generations



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="cwq", help='cwq or webqsp')
    parser.add_argument('--retrieve_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to retrieve.')
    parser.add_argument('--generate_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to generate.')
    parser.add_argument('--retrieve_samples', type=float, default=1., help='number of retriever sampling')
    parser.add_argument('--generate_samples', type=float, default=1., help='number of generation sampling')
    parser.add_argument('--examples', type=int, default=100,)
    parser.add_argument('--test', action='store_true', help="test or train.")

    args = parser.parse_args()
    args.retrieve_samples = int(args.retrieve_samples)
    args.generate_samples = int(args.generate_samples)
    if args.test:
        args.examples = 500

    wandb.init(project="KGQA", config=args)
    args = wandb.config

    retrieve_model_name = args.retrieve_model_name
    generate_model_name = args.generate_model_name

    num_retrieve_samples = args.retrieve_samples
    num_generate_samples = args.generate_samples

    ret_model_size = model_size_mapping[retrieve_model_name]
    gen_model_size = model_size_mapping[generate_model_name]

    # if ret_model_size != 32 and gen_model_size != 8:
    #     wandb.finish()
    #     print('no ret 32b and no gen 8b!!!')
    #     sys.exit(0)
    #
    # if ret_model_size >=70 and num_retrieve_samples > 30:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)
    #
    # if ret_model_size >= 30 and num_retrieve_samples > 50:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)
    #
    # if ret_model_size >= 7 and num_retrieve_samples > 70:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)
    #
    # if gen_model_size >=70 and num_generate_samples > 30:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)
    #
    # if gen_model_size >=30 and num_generate_samples > 50:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)
    #
    # if gen_model_size >= 7 and num_generate_samples > 70:
    #     wandb.finish()
    #     print('budget is not enough!!!')
    #     sys.exit(0)


    num_examples = args.examples
    task = args.task
    print('task:', task)
    print("retrieve model name:", retrieve_model_name)
    print("generate model name:", generate_model_name)
    print("number of retrieve samples:", num_retrieve_samples)
    print("number of generate samples:", num_generate_samples)

    archon_config = {
        "name": "archon-kgqa",
        "custom": True,  # SET THE CONFIG TO USE CUSTOM
        "layers": [
            [
                {
                    "type": "retrieve_component",  # custom type here
                    "model": retrieve_model_name,
                    "model_type": "custom_qwen",  # which api or which Generator
                    "temperature": 0.9,
                    "max_tokens": 512,
                    "samples": num_retrieve_samples,
                },
            ],
            [
                {
                    "type": "retrieve_merge",  # custom type here
                    "model": "",
                    "model_type": "custom_qwen",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "samples": 1,
                }
            ],
            [
                {
                    "type": "retrieve_generate",
                    "model": generate_model_name,
                    "model_type": "custom_qwen",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "samples": num_generate_samples,
                }
            ],
            [
                {
                    "type": "fuser",
                    "model": generate_model_name,
                    "model_type": "custom_qwen",
                    "temperature": 0.3,
                    "max_tokens": 512,
                    "samples": 1
                }
            ]
        ],
    }

    # Start with custom config
    print('archon config:', archon_config)
    archon = Archon(archon_config)

    # Add component to Archon
    # name has to match config
    archon.add_component("retrieve_component", Retrieval)
    archon.add_component("retrieve_merge", Merge)
    archon.add_component("retrieve_generate", CustomGenerator)
    archon.add_generator("custom_qwen", custom_generator)

    # Have to manually initialize
    archon.initialize()

    # load data
    if args.test:
        with open(f"data/{task}/data_with_ct_0.5_test500.json", "r") as f:
            examples = json.load(f)
        examples = examples[:num_examples]
    else:
        with open(f"data/{task}/data_with_ct_0.5.json", "r") as f:
            examples = json.load(f)
        examples = examples[:num_examples]

    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    em_list = []
    ret_precision_list = []
    ret_recall_list = []
    ret_f1_list = []
    ret_acc_list = []
    for example in tqdm.tqdm(examples):
        if task == 'webqsp':
            question_field = 'RawQuestion'
        else:
            question_field = 'question'
        if print_log:
            print('\nlog question,', example[question_field])
        query = example[question_field]
        gt_triples = example["gt_triples"]
        crucial_triples = example["crucial_triples"]
        twohop_triples = example["2hop_triples"]
        answers = example["answer"]
        knowledge_triplets = []
        for triple in twohop_triples:
            if triple not in gt_triples:
                knowledge_triplets.append(triple)
        knowledge_triplets = knowledge_triplets[:100]
        knowledge_triplets.extend(crucial_triples)
        np.random.shuffle(knowledge_triplets)
        gt_ids = []
        for triple in crucial_triples:
            ind = knowledge_triplets.index(triple)
            gt_ids.append(ind+1)
        example['gt_ids'] = gt_ids
        example['knowledge_triplets'] = knowledge_triplets
        if print_log:
            print('\nlen(knowledge_triplets):', len(knowledge_triplets))

        # make conversation
        example['role'] = 'user'
        example['content'] = query
        testing_instruction = [example]
        try:
            response = archon.generate(testing_instruction)

            if print_log:
                print("log, response:", response)

            example['output'] = response
            gold_answers = example['answer']
            em, f1, precision, recall = metric_evaluation(response, gold_answers)
            if print_log:
                print('log resposne,', response)
                print('gold answers:', gold_answers)
                print('log gen f1,', f1)
                print('log gen precision,', precision)
                print('log gen recall,', recall)
                print('log gen em,', em)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
            selected_ids = example['selected_ids']
            ret_precision, ret_recall, ret_f1 = precision_recall_f1(gt_ids, selected_ids)
            ret_acc = accuracy(gt_ids, selected_ids)
            if print_log:
                print('log ret f1,', ret_f1)
                print('log ret precision,', ret_precision)
                print('log ret recall,', ret_recall)
                print('log ret acc,', ret_acc)

            ret_precision_list.append(ret_precision)
            ret_recall_list.append(ret_recall)
            ret_f1_list.append(ret_f1)
            ret_acc_list.append(ret_acc)
        except:
            continue

    # if generate_mode:
        # em_list, f1_list, precision_list, recall_list = evaluate(examples, "")
    print(f"Gen Exact Match: {np.mean(em_list)}")
    print(f"Gen F1 Score: {np.mean(f1_list)}")
    print(f"Gen Precision: {np.mean(precision_list)}")
    print(f"Gen Recall: {np.mean(recall_list)}")

    print(f"Ret Precision: {np.mean(ret_precision_list)}")
    print(f"Ret Recall: {np.mean(ret_recall_list)}")
    print(f"Ret F1: {np.mean(ret_f1_list)}")
    print(f"Ret ACC: {np.mean(ret_acc_list)}")
    wandb.log({"Gen_EM": np.mean(em_list), "Gen_F1": np.mean(f1_list), "Gen_Precision": np.mean(precision_list), "Gen_Recall": np.mean(recall_list),
               "Ret_F1": np.mean(ret_f1_list), "Ret_ACC": np.mean(ret_acc_list),
               "Ret_Precision": np.mean(ret_precision_list), "Ret_Recall": np.mean(ret_recall_list),})

    # wandb.log({"precision": np.mean(precision_list), "recall": np.mean(recall_list), "f1": np.mean(f1_list), 'accuracy': np.mean(acc_list)})
    wandb.finish()


'''
python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 2 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 2

test scripts for cwq:
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 10 --test > log_kgqa_cwq_test_1.log 2>&1 &
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 60 --test > log_kgqa_cwq_test_2.log 2>&1 &
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 2 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 15 --test > log_kgqa_cwq_test_3.log 2>&1 &
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 4 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 5 --test > log_kgqa_cwq_test_4.log 2>&1 &
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.1-70B-Instruct --generate_samples 10 --test > log_kgqa_cwq_test_5.log 2>&1 &
nohup python archon_kgqa.py --task cwq --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 50 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 30 --test > log_kgqa_cwq_test_6.log 2>&1 &

# best configuration: 
# ours, ret Qwen/Qwen2.5-72B-Instruct-AWQ 1 meta-llama/Llama-3.1-8B-Instruct 10, 0.776
# hpo, ret Qwen/Qwen2.5-72B-Instruct-AWQ 1 meta-llama/Llama-3.1-8B-Instruct 60, 0.784 
# mlcopilot, ret Qwen/Qwen2.5-72B-Instruct-AWQ 2 meta-llama/Llama-3.1-8B-Instruct 15 , 0.782
# zs, ret Qwen/Qwen2.5-72B-Instruct-AWQ 4 meta-llama/Llama-3.1-8B-Instruct 5 , 0.764
# bayes, ret Qwen/Qwen2.5-7B-Instruct-AWQ 1 meta-llama/Llama-3.1-70B-Instruct 10, 0.778
# random, ret Qwen/Qwen2.5-7B-Instruct-AWQ 50 meta-llama/Llama-3.1-8B-Instruct 30, 0.784


test scripts for webqsp:
nohup python archon_kgqa.py --task webqsp --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 50 --test > log_kgqa_webqsp_test_1.log 2>&1 &
nohup python archon_kgqa.py --task webqsp --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.1-70B-Instruct --generate_samples 4 --test > log_kgqa_webqsp_test_2.log 2>&1 &
nohup python archon_kgqa.py --task webqsp --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 30 --generate_model_name meta-llama/Llama-3.1-70B-Instruct --generate_samples 10 --test > log_kgqa_webqsp_test_3.log 2>&1 &
nohup python archon_kgqa.py --task webqsp --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 10 --generate_model_name meta-llama/Llama-3.1-70B-Instruct --generate_samples 10 --test > log_kgqa_webqsp_test_4.log 2>&1 &

# best config: 
# ours: ret Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.1-8B-Instruct 50 0.894
# hpo: ret Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.1-8B-Instruct 50  0.894
# mlcopilot: ret Qwen/Qwen2.5-7B-Instruct-AWQ 1, gen meta-llama/Llama-3.1-70B-Instruct 4 0.88
# zs: ret Qwen/Qwen2.5-7B-Instruct-AWQ 30, gen meta-llama/Llama-3.1-70B-Instruct 10 0.885
# bayes: ret Qwen/Qwen2.5-7B-Instruct-AWQ 10, gen meta-llama/Llama-3.1-70B-Instruct 10 0.880
# random: ret Qwen/Qwen2.5-7B-Instruct-AWQ 10, gen meta-llama/Llama-3.1-70B-Instruct 10 0.880

'''