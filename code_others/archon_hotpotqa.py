from archon.completions.utils import custom_generator, custom_vision_generator
from archon.completions import Archon # 'Archon' is the initialization class
from archon.completions.components import Component, Generator #

import os,sys
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
        query_references = conversation[-1]["content"]

        prompt = (f"You have been provided with a query and a set of references. "
                  f"Your task is to select references that comprehensively cover the information needed to answer the query. "
                  f"Your response should include the original query and a list of selected references in the following format: 'Query: [query]\nSelected references: [reference number] [reference details]'."
                  f"Here is the query and references: \n {query_references}.")
        # prompt = ("You are provided with a set of references related to a query. \
        # Your objective is to assess the accuracy and sufficiency of these references and select the most relevant ones for answering the query. \
        # Please format your response as follows: \n"
        #           "Query: [Insert the query here]\n"
        #           "Selected References: [reference number] [reference details]\n"
        #           f"{query_references}")
        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that return the beneficial references to the query.",
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

        if os.path.exists(retrieval_save_file):
            examples_with_key = {}
            with open(retrieval_save_file, 'r') as f:
                    for line in f:
                        example = json.loads(line)
                        examples_with_key[example['query']] = example
        else:
            examples_with_key = {}
        self.examples_with_key = examples_with_key
        # print(examples_with_key.keys())

        # Generator class helps us handle messaging
        # self.model = Generator(self.config, custom_generators=custom_components)
        # print(f"Retrieval initialized with model: {self.model_name}")

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

        query_references = conversation[-1]["content"]
        query = query_references.split('Query:')[1].split('References:')[0].strip()

        # first time, generate many candidates following Retrieval component and save to retrieval_save_file
        # second time, only use Merge component, load the candidates from retrieval_save_file files.
        assert len(candidates) > 0
        if len(candidates) == 0: # no use
            candidates = self.examples_with_key[query]['candidates']
            candidates = candidates[:self.samples]
        else:
            # save to local files
            references = query_references.split("References:")[1].strip()
            example = {
                "query": query,
                "references": references,  # str
                "candidates": candidates,
            }
            # if retrieval_save_file and save_ret: # is not None
            #     with open(retrieval_save_file, "a") as f:
            #         f.write(json.dumps(example) + '\n')


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
        candidate = candidates[0]
        selected_ids_str = ""
        for e in selected_ids:
            selected_ids_str += f"[{str(e)}]" + ", "
        generations = [candidate.split('Selected references:')[0] + f"\nSelected references: {selected_ids_str}"]
        if print_log: print('Merge results:', generations)
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
        query_references = conversation[-1]["content"]
        query = query_references.split('Query:')[1].split('References:')[0].strip()
        references = query_references.split("References:")[1].strip().split('\n')
        prompts = []
        example = conversation[-1]
        selected_references_ids = candidates[0].split('Selected references:')[1]
        selected_references_ids = re.findall(r'\[(\d+)\]', selected_references_ids)  # string list
        example['merged_retrieval_response'] = [int(e)-1 for e in selected_references_ids]
        if len(candidates) > 0: # prompts from previous candidates
            # if it is from Merge, len(candidates) is 1
            for candidate in candidates:
                # candidate = candidates[0]
                selected_references_ids = candidate.split('Selected references:')[1]
                selected_references_ids = re.findall(r'\[(\d+)\]', selected_references_ids)  # string list
                selected_references = []
                for reference in references:
                    for rid in selected_references_ids:
                        if f"[{rid}]" in reference:
                            reference_new = reference.split(']')[1].strip()
                            selected_references.append(reference_new)
                            continue
                query_references = "Query: " + query + "\nReferences:\n"
                for i, reference in enumerate(selected_references):
                    query_references += f"[{i+1}] " + reference + "\n"

                prompt = (f"You will be provided with a set of references related to the query. "
                          f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
                          f"Here is the query and references: \n {query_references}.")
                prompts.append(prompt)
            prompt = prompts[0]
            if print_log: print('Log, prompt:', prompt)
        else: # prompts from conversation
            query_references = conversation[-1]["content"]
            prompt = (f"You will be provided with a set of references related to the query. "
                      f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
                      f"Here is the query and references: \n {query_references}.")
            # print('Log, prompt:', prompt)
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

        return generations



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_retrieve', action='store_true', help="Do not retrieve references.")
    parser.add_argument('--retrieve_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to retrieve.')
    parser.add_argument('--generate_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to generate.')
    parser.add_argument('--retrieve_samples', type=float, default=1., help='number of retriever sampling')
    parser.add_argument('--generate_samples', type=float, default=1., help='number of generation sampling')
    parser.add_argument('--merge_existing', action='store_true', help='Only using merge to see the performance with the amount of sampling.')
    parser.add_argument("--merge_samples", type=int, default=1,)
    parser.add_argument('--generate', action='store_true', help="set the config to generate anwers. ")
    parser.add_argument('--save_ret', action='store_true', help='whether saving to retrieval_save_file file.')
    parser.add_argument('--retrieval_save_file', type=str, default="retrieval_file.jsonl")
    parser.add_argument('--examples', type=int, default=50,)
    parser.add_argument('--test', action='store_true', help="test or train.")

    args = parser.parse_args()
    retrieve_model_name = args.retrieve_model_name
    generate_model_name = args.generate_model_name
    args.retrieve_samples = int(args.retrieve_samples)
    args.generate_samples = int(args.generate_samples)
    if args.test:
        args.examples = 500

    wandb.init(project="retrieve-and-generate-hotpotqa", config=args)
    args = wandb.config

    num_retrieve_samples = args.retrieve_samples
    num_generate_samples = args.generate_samples

    if retrieve_model_name == generate_model_name:
        wandb.finish()
        print('Two models cannot be the same.')
        sys.exit(0)

    # if '72b' in retrieve_model_name.lower():
    #     if num_retrieve_samples > 10:
    #         wandb.finish()
    #         print('budget is not enough!!!')
    #         sys.exit(0)
    # elif '32b' in retrieve_model_name.lower():
    #     if num_retrieve_samples > 15:
    #         wandb.finish()
    #         print('budget is not enough!!!')
    #         sys.exit(0)
    #
    # if '70b' in generate_model_name.lower():
    #     if num_generate_samples > 10:
    #         wandb.finish()
    #         print('budget is not enough!!!')
    #         sys.exit(0)

    num_examples = args.examples
    merge_samples = args.merge_samples
    merge_existing = args.merge_existing
    save_ret = args.save_ret
    no_retrieve = args.no_retrieve
    generate_mode = args.generate
    retrieval_save_file = args.retrieval_save_file
    print("retrieve model name:", retrieve_model_name)
    print("generate model name:", generate_model_name)
    print("number of retrieve samples:", num_retrieve_samples)
    print("number of generate samples:", num_generate_samples)
    print("merge samples:", merge_samples)  # only for the time saving for test-time sampling
    print("merge existing:", merge_existing)
    print("generate mode:", generate_mode)
    print("no_retrieve:", no_retrieve)
    print("retrieval saved file:", retrieval_save_file)

    if merge_existing:
        assert os.path.exists(retrieval_save_file)
        archon_config = {
            "name": "archon-retrieve",
            "custom": True,  # SET THE CONFIG TO USE CUSTOM
            "layers": [
                [
                    {
                        "type": "retrieve_merge",  # custom type here
                        "model": "",
                        "model_type": "custom_qwen",
                        "temperature": 0.,
                        "max_tokens": 512,
                        "samples": merge_samples,
                    },
                ]
            ],
        }
    elif generate_mode: #
        if no_retrieve:
            if num_generate_samples == 1:
                archon_config = {
                    "name": "archon-retrieve-generate",
                    "custom": True,  # SET THE CONFIG TO USE CUSTOM
                    "layers": [
                        [
                            {
                                "type": "retrieve_generate",
                                "model": generate_model_name,
                                "model_type": "custom_qwen",
                                "temperature": 0.3,
                                "max_tokens": 512,
                                "samples": num_generate_samples,
                            }
                        ]
                    ],
                }
            else:  # num_samples > 1
                archon_config = {
                    "name": "archon-retrieve-generate",
                    "custom": True,  # SET THE CONFIG TO USE CUSTOM
                    "layers": [
                        [
                            {
                                "type": "retrieve_generate",
                                "model": generate_model_name,
                                "model_type": "custom_qwen",
                                "temperature": 0.9,
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
                                "max_tokens": 1024,
                                "samples": 1
                            }
                        ]
                    ],

                }

        else: # retrieve-merge-generate
            archon_config = {
                "name": "archon-retrieve-generate",
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
    elif num_retrieve_samples > 1: # retrieve and merge mode
        archon_config = {
            "name": "archon-retrieve",
            "custom": True,  # SET THE CONFIG TO USE CUSTOM
            "layers": [
                [
                    {
                        "type": "retrieve_component",  # custom type here
                        "model": retrieve_model_name,
                        "model_type": "custom_qwen",
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
            ],
        }
    else: # no merge, just retrieve once, num_retrieve_samples = 1
        archon_config = {
            "name": "archon-retrieve",
            "custom": True,  # SET THE CONFIG TO USE CUSTOM
            "layers": [
                [
                    {
                        "type": "retrieve_component",  # custom type here
                        "model": retrieve_model_name,
                        "model_type": "custom_qwen",
                        "temperature": 0.9,
                        "max_tokens": 512,
                        "samples": 1,
                    },
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
    if not args.test:
        with open("data/HotpotQA/tts_test.json", "r") as f:
            examples = json.load(f)
        examples = examples[:num_examples]
    else:
        with open("data/HotpotQA/tts_test_2_samples500.json", "r") as f:
            examples = json.load(f)
        print('Len of test examples:', len(examples))
        examples = examples

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
        if print_log: print('log question,', example['question'])
        query = "Query: " + example['question']
        references = example['ctxs_candidate']
        ctxs_gt = example['ctxs_gt']
        ctxs_gt_ids = []
        for ctx in ctxs_gt:
            id = references.index(ctx)
            ctxs_gt_ids.append(id)
        query += "References: "
        for i, reference in enumerate(references):
            query += f'\n[{i+1}] ' + reference
        # print(query)
        # we can extract the question from the query, and reference list from the query.

        # make conversation
        example['role'] = 'user'
        example['content'] = query
        testing_instruction = [example]

        response = archon.generate(testing_instruction)
        if generate_mode:
            example['output'] = response
            gold_answer = example['answer']
            em, f1, precision, recall = metric_evaluation(response, [gold_answer])
            if print_log:
                print('log resposne,', response)
                print('gold answers:', gold_answer)
                print('log gen f1,', f1)
                print('log gen precision,', precision)
                print('log gen recall,', recall)
                print('log gen em,', em)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
            retrieval_reference_ids = example['merged_retrieval_response']
            ret_precision, ret_recall, ret_f1 = precision_recall_f1(ctxs_gt_ids, retrieval_reference_ids)
            ret_acc = accuracy(ctxs_gt_ids, retrieval_reference_ids)
            if print_log:
                print('log ret f1,', ret_f1)
                print('log ret precision,', ret_precision)
                print('log ret recall,', ret_recall)
                print('log ret acc,', ret_acc)

            ret_precision_list.append(ret_precision)
            ret_recall_list.append(ret_recall)
            ret_f1_list.append(ret_f1)
            ret_acc_list.append(ret_acc)
        else:
            try:
                reference_ids = re.findall(r'\[(\d+)\]', response)
                reference_ids = [int(e)-1 for e in reference_ids]
            except:
                reference_ids = []
            print('log, reference ids:', reference_ids)
            print('log, gt ids:', ctxs_gt_ids)
            precision, recall, f1 = precision_recall_f1(ctxs_gt_ids, reference_ids)
            acc = accuracy(ctxs_gt_ids, reference_ids)
            print('log, precision', precision)
            print('recall', recall)
            print('f1', f1)
            print('acc ', acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            acc_list.append(acc)

    if generate_mode:
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
    else:
        print('mean precision:', np.mean(precision_list))
        print('mean recall:', np.mean(recall_list))
        print('mean f1:', np.mean(f1_list))
        print('mean accuracy:', np.mean(acc_list))
        wandb.log({"precision": np.mean(precision_list), "recall": np.mean(recall_list), "f1": np.mean(f1_list), 'accuracy': np.mean(acc_list)})
    wandb.finish()


'''
example script: 
python archon_hotpotqa.py --generate --retrieve_model_name Qwen/Qwen2.5-7B-Instruct-AWQ --retrieve_samples 2 --generate_model_name meta-llama/Llama-3.1-8B-Instruct --generate_samples 2

test script: 
nohup python archon_hotpotqa.py --generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 40 --test > log_test_hotpotqa_1.log 2>&1 &
nohup python archon_hotpotqa.py --generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 30 --test > log_test_hotpotqa_2.log 2>&1 &
nohup python archon_hotpotqa.py --generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 25 --test > log_test_hotpotqa_3.log 2>&1 &
nohup python archon_hotpotqa.py --generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 10 --test > log_test_hotpotqa_4.log 2>&1 &

# best config: 
# ours: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 40, 0.742 0.722
# hpo: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 40, 0.742 0.722
# mlcopilot: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 25, 0.708 0.724
# llmzs: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 10 0.714
# bayes: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 30 0.716 0.706
# random: ret, Qwen/Qwen2.5-72B-Instruct-AWQ 1, gen meta-llama/Llama-3.2-3B-Instruct 30 0.716 0.706
'''

