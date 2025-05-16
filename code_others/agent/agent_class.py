from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pandas as pd
import numpy as np
import re
import json
import os
from openai import OpenAI

openai_api_key="your_api_key"

# model_size_mapping = {"Qwen/Qwen2.5-7B-Instruct-AWQ":7, "Qwen/Qwen2.5-32B-Instruct-AWQ":32, "Qwen/Qwen2.5-72B-Instruct-AWQ":72,
# "meta-llama/Llama-3.2-3B-Instruct": 3, "meta-llama/Llama-3.1-8B-Instruct": 8, "meta-llama/Llama-3.1-70B-Instruct": 70, "google/gemma-2-2b-it": 2, "google/gemma-2-9b-it": 9, "google/gemma-2-27b-it": 27, "microsoft/Phi-3-mini-4k-instruct": 3.8, "microsoft/Phi-3-small-8k-instruct": 7, "microsoft/Phi-3-medium-4k-instruct": 14, "infly/OpenCoder-1.5B-Instruct": 1.5, "infly/OpenCoder-8B-Instruct": 8
# }

model_specs = {
    "Qwen/Qwen2.5-7B-Instruct-AWQ": {"M": 7_000_000_000, "L": 32, "D": 4_096},
    "Qwen/Qwen2.5-32B-Instruct-AWQ": {"M": 32_000_000_000, "L": 48, "D": 6_144},
    "Qwen/Qwen2.5-72B-Instruct-AWQ": {"M": 72_000_000_000, "L": 80, "D": 8_192},
    "meta-llama/Llama-3.2-3B-Instruct": {"M": 3_000_000_000, "L": 32, "D": 3_072},
    "meta-llama/Llama-3.1-8B-Instruct": {"M": 8_000_000_000, "L": 32, "D": 4_096},
    "meta-llama/Llama-3.1-70B-Instruct": {"M": 70_000_000_000, "L": 80, "D": 8_192},
    "llama3b": {"M": 3_000_000_000, "L": 32, "D": 3_072},
    "llama8b": {"M": 8_000_000_000, "L": 32, "D": 4_096},
    "llama70b": {"M": 70_000_000_000, "L": 80, "D": 8_192},
    "google/gemma-2-2B-it": {"M": 2_000_000_000, "L": 28, "D": 2_048},
    "google/gemma-2-9B-it": {"M": 9_000_000_000, "L": 32, "D": 4_096},
    "google/gemma-2-27B-it": {"M": 27_000_000_000, "L": 40, "D": 6_144},
    "microsoft/Phi-3-mini-4k-instruct": {"M": 3_800_000_000, "L": 32, "D": 3_072},
    "microsoft/Phi-3-small-8k-instruct": {"M": 7_000_000_000, "L": 32, "D": 4_096},
    "microsoft/Phi-3-medium-4k-instruct": {"M": 14_000_000_000, "L": 40, "D": 5_120},
    "infly/OpenCoder-1.5B-Instruct": {"M": 1_500_000_000, "L": 24, "D": 2_048},
    "infly/OpenCoder-8B-Instruct": {"M": 8_000_000_000, "L": 32, "D": 4_096},
}

class Task(object):
    def __init__(self, name, desc, budget, model_choices, pattern, sub_tasks, agent):
        self.name = name
        self.desc = desc
        self.budget = budget
        self.model_choices = model_choices
        self.pattern = pattern
        self.sub_tasks = sub_tasks
        self.agent = agent

    def get_tts_strategy(self, iterations, batch_size):
        return self.agent.run(iterations, batch_size, trace=True)

class AgentTTS(object):
    def __init__(self, task_name, task_desc, budget, model_choices, pattern, sub_tasks, Np_list, Nd_list, llm, example=None):
        self.task_name = task_name
        self.task_desc = task_desc
        self.budget = budget
        self.model_choices = model_choices
        self.pattern = pattern
        self.sub_tasks = sub_tasks
        self.Np_list = Np_list
        self.Nd_list = Nd_list
        self.archive = Archive(task_name)
        self.llm = llm
        self.example = example

        self.sub_tasks_dict = {}
        for subtask, model_list, Np, Nd in zip(sub_tasks, model_choices, Np_list, Nd_list):
            largest = 0
            for model in model_list:
                model_size = self.get_model_size(pattern, model)
                if model_size > largest:
                    largest = model_size
            self.sub_tasks_dict[subtask] = {}
            self.sub_tasks_dict[subtask]['largest_model_size'] = largest
            self.sub_tasks_dict[subtask]['Np'] = Np
            self.sub_tasks_dict[subtask]['Nd'] = Nd


        self.M_dict = {}
        for subtask, model_list in zip(sub_tasks, model_choices):
            largest = 0
            for model in model_list:
                model_size = self.get_model_size(pattern, model)
                if model_size > largest:
                    largest = model_size
            self.M_dict[subtask] = largest
        print('M dict:', self.M_dict)

        assert len(model_choices) == len(sub_tasks)

    def run(self, iterations: int, batch_size: int, trace=True):
        main_metric = self.archive.environment.main_metric
        # step 1. init candidate, update history
        new_candidates = self.init_candidates()
        print('### initial candidates:', new_candidates, '\n\n\n\n')
        new_candidates, scores = self.get_scores(new_candidates)  # scores are feedback
        print('Log, scores for initial candidates:', scores)

        recent_added = 0
        for candidate, score in zip(new_candidates, scores):
            print('### new candidate:', candidate)
            print('### score:', score)
            cost = self.calculate_budget(candidate)
            if score and score[main_metric] > 0:
                self.update_archive_history(candidate, score, cost)
                recent_added += 1

        for iteration in range(iterations):
            print(f"\n\nIteration {iteration + 1} - Generated candidates:\n{json.dumps(new_candidates, indent=2)}")

            # step 2. gen new experience, serve as further feedback
            if iteration == 0:
                init = True
            else:
                init = False
            new_experience = self.gen_new_experience(recent_added_samples=recent_added, init=init)
            self.update_experience(new_experience)

            # step 3. gen new candidates, update history
            new_candidates = self.gen_new_candidates(batch_size)
            new_candidates, scores = self.get_scores(new_candidates)  # list[dict]
            recent_added = 0
            for candidate, score in zip(new_candidates, scores):
                print('### new candidate:', candidate)
                print('### score:', score)
                cost = self.calculate_budget(candidate)
                if cost > self.budget:
                    print('Budget exceeded')
                    continue

                if score and score[main_metric] > 0:
                    self.update_archive_history(candidate, score, cost)
                    recent_added += 1
                    print(f"Score: {score[main_metric]:.2f} | Cost: {cost:.2f} | Params: {candidate}")
            print("### Log, len(history):)", len(self.archive.history))
            # step 4. until the end

        # get the best score
        history = self.get_evaluation_results()
        main_metric = self.get_main_metric()
        results = []
        print("history: ")
        for i, item in enumerate(history):
            params = item["params"]
            main_score = item["score"][main_metric]
            cost = item["cost"]
            results.append((i, params, main_score, cost))
            print({'i': i, 'params': params, 'main_score': main_score, 'cost': cost})

        if trace:
            best_main_score = 0
            trace = []
            for step, item in enumerate(history):
                main_score = item["score"][main_metric]
                if main_score > best_main_score:
                    best_main_score = main_score
                    trace.append((step, best_main_score, item))
            return results, trace



        print("\nFinal archive (sorted by score):")
        for i, params, score, cost in sorted(results, key=lambda x: -x[2]):
            print(f"[{score:.2f}] {params} Used budget {cost}, index {i}")

        return results


    def update_experience(self, new_experience):
        self.archive.experience.append(new_experience)

    def get_scores(self, new_candidates):
        new_candidates, scores = self.archive.evaluate_batch(new_candidates, self.budget)
        return new_candidates, scores

    def update_archive_history(self, new_candidate, score, cost):
        self.archive.add_new_params(new_candidate, score, cost)

    def get_evaluation_results(self):
        return self.archive.history

    def get_main_metric(self):
        return self.archive.environment.main_metric

    @abstractmethod
    def gen_new_candidates(self, batch_size):
        '''
        使用archive里面的历史和上一次的经验来生成新的候选，生成候选的策略
        # (1) 突变采样数量，increase/decrease the samples
        # (2) 突变模型大小
        # (3) 交叉选择参数
        # (4) 随机生成新的参数组合
        设计内部函数，包括突变，交叉选择参数，和生成新的参数组合
        :param init:
        :return:
        '''
        pass

    @abstractmethod
    def gen_new_experience(self, recent_added_samples:int, init=False):
        pass

    @abstractmethod
    def init_candidates(self):
        pass

    def get_model_size(self, pattern, model_name):
        # r = re.findall(pattern, model_name)
        # if len(r) != 0:
        #     model_size = int(r[0])
        # else:
        #     model_size = 0
        model_size = model_specs[model_name]["M"]
        return model_size

    def extract_json(self, text: str) -> dict:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def compute_budget(self, S, M_large, Np1, Nd1, M_small=3e9, Np2=128, Nd2=64):
        """
        Compute the FLOPs-based budget of generating S samples on a large model,
        normalized to the FLOPs of a small model's unit budget.

        Parameters:
        - S: number of samples generated by the large model
        - M_small: parameter count of the small model (e.g., 3B, ignore B)
        - M_large: parameter count of the large model (e.g., 70B)
        - Np2: prompt length for small model (unit config)
        - Nd2: decode length for large model (unit config)
        - Np1: prompt length for the new config
        - Nd1: decode length for the new config

        Returns:
        - budget: normalized compute budget
        """
        alpha = M_large / M_small
        beta1 = Np1 / Nd1
        beta2 = Np2 / Nd2  # normalize to same decode length as unit config
        beta3 = Np1 / Np2

        budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1)
        return budget

    def calculate_budget(self, params):
        def subunit_cost(subtask: str, model: str, samples: int) -> float:
            # M = self.M_dict[subtask]
            m = self.get_model_size(self.pattern, model)
            samples = samples
            Np = self.sub_tasks_dict[subtask]['Np']
            Nd = self.sub_tasks_dict[subtask]['Nd']

            budget = self.compute_budget(samples, m, Np, Nd)
            return budget

            # denominator = 4 * (M / m) - 3
            # return samples / denominator if denominator != 0 else float('inf')

        return sum(
            subunit_cost(t, p["model"], p["samples"])
            for t, p in params.items()
        )

    def check_budget(self, candidate: Dict):
        print(candidate, 'budget:', self.calculate_budget(candidate))
        return self.calculate_budget(candidate) <= self.budget


class LLM(ABC):

    @abstractmethod
    def generate(self, prompt, system_prompt):
        pass

class ChatGPT(LLM):
    def __init__(self, model_name=None):
        self.api_key = openai_api_key
        if model_name is None:
            self.model_name = "o3-mini-2025-01-31"
        else:
            self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            reasoning_effort="high"
        )
        answer = response.choices[0].message.content
        # print('\nLog, ChatGPT, response:', answer)
        return answer







class Archive(object):
    def __init__(self, task_name):
        self.task_name = task_name
        self.history = []
        self.experience = []
        self.environment = Environment(task_name)

    def evaluate_batch(self, batch, budget):
        results = []
        new_parameters = []
        for parameter in batch:
            # print("before search:", parameter)
            new_parameter, result = self.environment.run(parameter, budget)
            # print("after search:", new_parameter, result)
            if new_parameter is not None and result is not None:
                results.append(result)
                new_parameters.append(new_parameter)
        return new_parameters, results

    def add_new_params(self, param, score, cost):
        self.history.append({"params": param, "score": score, "cost": cost})


#
#
# def calculate_budget_ret_gen(row, max_size1=72, max_size2=70):
#     def calculate_unit_retrieve_model(r, max_size):
#         unit_retrieve_model = 4 * (max_size // r) - 3
#         return unit_retrieve_model
#
#     def calculate_unit_generate_model(g, max_size):
#         unit_generate_model = 4 * (max_size // g) - 3
#         return unit_generate_model
#
#     for i in range(max_size1, 1, -1):
#         if f"{i}b" in row['retrieve_model_name'].lower():
#             r_model = i
#             break
#
#     for i in range(max_size2, 1, -1):
#         if f"{i}b" in row['generate_model_name'].lower():
#             g_model = i
#             break
#
#     r_samples = int(row['retrieve_samples']) if pd.notna(row['retrieve_samples']) else 0
#     g_samples = int(row['generate_samples']) if pd.notna(row['generate_samples']) else 0
#
#     unit_retrieve_model = calculate_unit_retrieve_model(r_model, max_size1)
#     unit_generate_model = calculate_unit_generate_model(g_model, max_size2)
#
#     budget_usage = r_samples / unit_retrieve_model + g_samples / unit_generate_model
#     return budget_usage


def compute_budget(S, M_large, Np1, Nd1, M_small=3e9, Np2=128, Nd2=64):
    alpha = M_large / M_small
    beta1 = Np1 / Nd1
    beta2 = Np2 / Nd2  # normalize to same decode length as unit config
    beta3 = Np1 / Np2

    budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1)
    return budget


def calculate_budget_ret(row, Np, Nd):
    r_model = model_specs[row['retrieve_model_name']]['M']
    # alpha = 72 / r_model
    r_samples = int(row['retrieve_samples']) if pd.notna(row['retrieve_samples']) else 0
    # Np = 2048
    # Nd = 128
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget


def calculate_budget_gen(row, Np, Nd):
    g_model = model_specs[row['generate_model_name']]['M']
    g_samples = int(row['generate_samples']) if pd.notna(row['generate_samples']) else 0

    # budget = g_samples/alpha - (1- 1/alpha)*beta
    # Np = 256
    # Nd = 64
    budget = compute_budget(g_samples, g_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget


def calculate_budget_ret_gen_2wiki(row):
    Np_list = [2048, 256]
    Nd_list = [128, 64]
    budget_ret = calculate_budget_ret(row, Np_list[0], Nd_list[0])
    budget_gen = calculate_budget_gen(row, Np_list[1], Nd_list[1])
    budget = budget_ret + budget_gen

    return budget


def calculate_budget_ret_gen_hotpot(row):
    Np_list = [2048, 256]
    Nd_list = [128, 64]
    budget = calculate_budget_ret_gen_2wiki(row)
    return budget


def calculate_budget_ret_gen_cwq(row):
    Np_list = [1024, 256]
    Nd_list = [64, 64]
    budget_ret = calculate_budget_ret(row, Np_list[0], Nd_list[0])
    budget_gen = calculate_budget_gen(row, Np_list[1], Nd_list[1])
    # print('###', budget_ret, budget_gen)
    budget = budget_ret + budget_gen

    return budget

def calculate_budget_ret_gen_webqsp(row):
    Np_list = [1024, 256]
    Nd_list = [64, 64]
    budget = calculate_budget_ret_gen_cwq(row)

    return budget



def calculate_budget_decomp(row, Np, Nd):
    r_model = model_specs[row['decomposer_model_name']]['M']
    r_samples = int(row['decompose_samples']) if pd.notna(row['decompose_samples']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget


def calculate_budget_tool(row, Np, Nd):
    r_model = model_specs[row['api_selection_model_name']]['M']
    r_samples = int(row['api_selection_samples']) if pd.notna(row['api_selection_samples']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget

def calculate_budget_parameter(row, Np, Nd):
    r_model = model_specs[row['parameter_selection_model_name']]['M']
    r_samples = int(row['parameter_selection_samples']) if pd.notna(row['parameter_selection_samples']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget

def calculate_budget_taskbench(row):
    Np_list = [1024, 1024, 1024]
    Nd_list = [64, 256, 2048]
    budget_decomp = calculate_budget_decomp(row, Np_list[0], Nd_list[0])
    budget_tool = calculate_budget_tool(row, Np_list[1], Nd_list[1])
    budget_parameter = calculate_budget_parameter(row, Np_list[2], Nd_list[2])
    # print('###', budget_ret, budget_gen)
    budget = budget_decomp + budget_tool + budget_parameter

    return budget


def calculate_budget_code(row, Np, Nd):
    r_model = model_specs[row['coding_model']]['M']
    r_samples = int(row['coding_sampling']) if pd.notna(row['coding_sampling']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget
def calculate_budget_review(row, Np, Nd):
    r_model = model_specs[row['review_model']]['M']
    r_samples = int(row['review_sampling']) if pd.notna(row['review_sampling']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget
def calculate_budget_test(row, Np, Nd):
    r_model = model_specs[row['test_model']]['M']
    r_samples = int(row['test_sampling']) if pd.notna(row['test_sampling']) else 0
    budget = compute_budget(r_samples, r_model, Np, Nd)
    if budget < 0:
        budget = 0
    return budget

def calculate_budget_chatdev(row):
    Np_list = [1024, 1024, 1024]
    Nd_list = [1024, 512, 256]
    budget_code = calculate_budget_code(row, Np_list[0], Nd_list[0])
    budget_review = calculate_budget_review(row, Np_list[1], Nd_list[1])
    budget_test = calculate_budget_test(row, Np_list[2], Nd_list[2])
    # print('###', budget_ret, budget_gen)
    budget = budget_code + budget_review + budget_test

    return budget


class Environment(object):
    def __init__(self, task_name):
        self.task_name = task_name
        print('Task name in Environment:', task_name)
        if task_name == "2wikihopqa":
            df = pd.read_csv('CSV_files/ret_then_gen_2wiki.csv')
            # df = pd.concat([df1, df2])
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_2wiki, axis=1)
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        if task_name == "2wikihopqa_100":
            df = pd.read_csv('CSV_files/ret_then_gen_2wiki_100.csv')
            # df = pd.concat([df1, df2])
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_2wiki, axis=1)
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        if task_name == "2wikihopqa_75":
            df = pd.read_csv('CSV_files/ret_then_gen_2wiki_75.csv')
            # df = pd.concat([df1, df2])
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_2wiki, axis=1)
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        elif task_name == "hotpotqa":
            df = pd.read_csv('CSV_files/ret_then_gen_hotpotqa2.csv')
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_hotpot, axis=1)
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        elif task_name == "cwq":
            df = pd.read_csv("./CSV_files/kgqa_wandb_cwq.csv")
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_cwq, axis=1)
            # print(df.head().to_dict('records'))
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        elif task_name == "webqsp":
            df = pd.read_csv("./CSV_files/kgqa_wandb_webqsp.csv")
            df = df[['examples', 'generate_model_name', 'generate_samples', 'retrieve_model_name', 'retrieve_samples',
                     'Gen_EM', 'Gen_F1', 'Gen_Precision', 'Gen_Recall', 'Ret_ACC', 'Ret_F1', 'Ret_Precision',
                     'Ret_Recall']]
            df['budget'] = df.apply(calculate_budget_ret_gen_webqsp, axis=1)
            self.metrics = ['Gen_EM', 'Ret_F1']
            self.main_metric = 'Gen_EM'
        elif task_name == "taskbench_dailylifeapis":
            df = pd.read_csv("./CSV_files/taskbench_dailylifeapis.csv")
            df = df[["decomposer_model_name", "decompose_samples", "api_selection_model_name", "api_selection_samples", "parameter_selection_model_name", "parameter_selection_samples", "step_rougeL", "step_bertscore_f1", "node_f1", "argument_f1", "unsuccess_num"]]
            df['budget'] = df.apply(calculate_budget_taskbench, axis=1)
            self.metrics = ['step_rougeL', 'node_f1', 'argument_f1']
            self.main_metric = 'argument_f1'
        elif task_name == "chatdev":
            df = pd.DataFrame()
            p = "./CSV_files/"
            for f in os.listdir(p):
                if os.path.isfile(os.path.join(p, f)) and f.startswith('chatdev_test'):
                    df_s = pd.read_csv(os.path.join(p, f))
                    df = pd.concat([df, df_s], ignore_index=True)
            df = df[
                ['coding_model', 'coding_sampling', 'review_model', 'review_sampling', 'test_model', 'test_sampling',
                 'mean_completeness', 'mean_consistency', 'mean_executability', 'unsuccess_count']]

            df['budget'] = df.apply(calculate_budget_chatdev, axis=1)
            self.metrics = ['mean_consistency']
            self.main_metric = 'mean_consistency'
        else:
            pass
        self.db = df

    def run(self, params, budget):

        if self.task_name in ["2wikihopqa","2wikihopqa_75","2wikihopqa_100", "hotpotqa", "cwq", "webqsp"]:
            def convert_row_to_params(row):
                retrieve_model = row['retrieve_model_name']
                retrieve_samples_target = row['retrieve_samples']
                generate_model = row['generate_model_name']
                generate_samples_target = row['generate_samples']
                params = {}
                params['Retrieval'] = {}
                params['Question Answering'] = {}
                params['Retrieval']['model'] = retrieve_model
                params['Retrieval']['samples'] = retrieve_samples_target
                params['Question Answering']['model'] = generate_model
                params['Question Answering']['samples'] = generate_samples_target
                return params
            # items = self.db[
            #     (self.db['retrieve_model_name'] == params['Retrieval']['model']) &
            #     (self.db['retrieve_samples'] == params['Retrieval']['samples']) &
            #     (self.db['generate_model_name'] == params['Question Answering']['model']) &
            #     (self.db['generate_samples'] == params['Question Answering']['samples'])
            # ]
            items, diff = self.fuzzy_match(params, self.task_name, budget)
            items = items.to_dict(orient="records")
            # print("Log, diff", diff)
            if diff > 5:
                return None, None
        elif self.task_name == "taskbench_dailylifeapis":
            def convert_row_to_params(row):
                decomposer_model = row['decomposer_model_name']
                decomposer_samples_target = row['decompose_samples']
                api_selection_model = row['api_selection_model_name']
                api_selection_samples_target = row['api_selection_samples']
                parameter_selection_model = row['parameter_selection_model_name']
                parameter_selection_samples_target = row['parameter_selection_samples']
                params = {}
                params['Task Decomposition'] = {}
                params['Tool Selection'] = {}
                params['Parameter Prediction'] = {}
                params['Task Decomposition']['model'] = decomposer_model
                params['Task Decomposition']['samples'] = decomposer_samples_target
                params['Tool Selection']['model'] = api_selection_model
                params['Tool Selection']['samples'] = api_selection_samples_target
                params['Parameter Prediction']['model'] = parameter_selection_model
                params['Parameter Prediction']['samples'] = parameter_selection_samples_target
                return params
            items, diff = self.fuzzy_match(params, self.task_name, budget)
            items = items.to_dict(orient="records")
            # print("Log, diff", diff)
            if diff > 5:
                return None, None
        elif self.task_name == "chatdev":
            def convert_row_to_params(row):
                # 'coding_model', 'coding_sampling', 'review_model', 'review_sampling', 'test_model', 'test_sampling', 'mean_completeness', 'mean_consistency', 'mean_executability'
                coding_model = row['coding_model']
                coding_sampling = row['coding_sampling']
                review_model = row['review_model']
                review_sampling = row['review_sampling']
                test_model = row['test_model']
                test_sampling = row['test_sampling']
                params = {}
                params['Coding'] = {}
                params['Static Testing'] = {}
                params['Dynamic Testing'] = {}
                params['Coding']['model'] = coding_model
                params['Coding']['samples'] = coding_sampling
                params['Static Testing']['model'] = review_model
                params['Static Testing']['samples'] = review_sampling
                params['Dynamic Testing']['model'] = test_model
                params['Dynamic Testing']['samples'] = test_sampling
                return params
            items, diff = self.fuzzy_match(params, self.task_name, budget)
            items = items.to_dict(orient="records")
            # print("Log, diff", diff)
            if diff > 20:
                return None, None

        else:
            items = self.db[
                (self.db['review_model'] == params['review']['model']) &
                (self.db['review_sampling'] == params['review']['samples']) &
                (self.db['test_model'] == params['test']['model']) &
                (self.db['test_sampling'] == params['test']['samples']) &
                (self.db['coding_model'] == params['coding']['model']) &
                (self.db['coding_sampling'] == params['coding']['samples'])
                ].to_dict(orient="records")

        if len(items) == 1:
            return convert_row_to_params(items[0]), {metric: items[0][metric] for metric in self.metrics}
        elif len(items) > 1:
            return_dict = {}
            for metric in self.metrics:
                return_dict[metric]=np.max([item[metric] for item in items])
            return convert_row_to_params(items[0]), return_dict

        return None, None


    def fuzzy_match(self, params, task_name, budget=1000):
        if task_name in ["2wikihopqa","2wikihopqa_75","2wikihopqa_100", "hotpotqa", "cwq", "webqsp"]:
            # Extract parameters
            retrieve_model = params['Retrieval']['model']
            retrieve_samples_target = params['Retrieval']['samples']
            generate_model = params['Question Answering']['model']
            generate_samples_target = params['Question Answering']['samples']

            # Filter by model names first
            filtered = self.db[
                (self.db['retrieve_model_name'] == retrieve_model) &
                (self.db['generate_model_name'] == generate_model) &
                (self.db['budget'] <= budget)
                ]
            # print('Log, filtered: ', filtered)

            # If filtered is empty early, skip the rest
            if filtered.empty:
                items = filtered
                min_diff = -1
            else:
                ret_model_size = model_specs[retrieve_model]["M"]
                gen_model_size = model_specs[generate_model]["M"]
                d1 = 72e9/ret_model_size * 4 - 3
                d2 = 70e9/gen_model_size * 4 - 3
                # Compute absolute difference for samples
                filtered = filtered.copy()  # avoid SettingWithCopyWarning
                filtered['retrieve_diff'] = np.abs(filtered['retrieve_samples'] - retrieve_samples_target)/d1
                filtered['generate_diff'] = np.abs(filtered['generate_samples'] - generate_samples_target)/d2
                filtered['total_diff'] = filtered['retrieve_diff'] + filtered['generate_diff']

                # Take the row with the smallest total difference
                min_diff = filtered['total_diff'].min()
                items = filtered[filtered['total_diff'] == min_diff]

        elif task_name == "taskbench_dailylifeapis":
            # Extract parameters
            decomposer_model=params['Task Decomposition']['model']
            decomposer_samples_target=params['Task Decomposition']['samples']
            api_selection_model=params['Tool Selection']['model']
            api_selection_samples_target=params['Tool Selection']['samples']
            parameter_selection_model=params['Parameter Prediction']['model']
            parameter_selection_samples_target=params['Parameter Prediction']['samples']

            # Filter by model names first
            filtered = self.db[
                (self.db['decomposer_model_name'] == decomposer_model) &
                (self.db['api_selection_model_name'] == api_selection_model) &
                (self.db['parameter_selection_model_name'] == parameter_selection_model) &
                (self.db['budget'] <= budget)
                ]
            # print('Log, filtered: ', filtered)

            # If filtered is empty early, skip the rest
            if filtered.empty:
                items = filtered
                min_diff = -1
            else:
                decomp_model_size = model_specs[decomposer_model]["M"]
                tool_model_size = model_specs[api_selection_model]["M"]
                parameter_model_size = model_specs[parameter_selection_model]["M"]
                d1 = 70e9 / decomp_model_size * 4 - 3
                d2 = 70e9 / tool_model_size * 4 - 3
                d3 = 70e9 / parameter_model_size * 4 - 3
                # Compute absolute difference for samples
                filtered = filtered.copy()  # avoid SettingWithCopyWarning
                filtered['decomp_diff'] = np.abs(filtered['decompose_samples'] - decomposer_samples_target) / d1
                filtered['tool_diff'] = np.abs(filtered['api_selection_samples'] - api_selection_samples_target) / d2
                filtered['parameter_diff'] = np.abs(filtered['parameter_selection_samples'] - parameter_selection_samples_target) / d3
                filtered['total_diff'] = filtered['decomp_diff'] + filtered['tool_diff'] + filtered['parameter_diff']

                # Take the row with the smallest total difference
                min_diff = filtered['total_diff'].min()
                items = filtered[filtered['total_diff'] == min_diff]
        elif task_name == "chatdev":
            # Extract parameters
            coding_model=params['Coding']['model']
            coding_sampling=params['Coding']['samples']
            review_model=params['Static Testing']['model']
            review_sampling=params['Static Testing']['samples']
            test_model=params['Dynamic Testing']['model']
            test_sampling=params['Dynamic Testing']['samples']

            # Filter by model names first
            filtered = self.db[
                (self.db['coding_model'] == coding_model) &
                (self.db['review_model'] == review_model) &
                (self.db['test_model'] == test_model) &
                (self.db['budget'] <= budget)
                ]
            # print('Log, filtered: ', filtered)

            # If filtered is empty early, skip the rest
            if filtered.empty:
                items = filtered
                min_diff = -1
            else:
                coding_model_size = model_specs[coding_model]["M"]
                review_model_size = model_specs[review_model]["M"]
                test_model_size = model_specs[test_model]["M"]
                d1 = 70e9 / coding_model_size * 4 - 3
                d2 = 70e9 / review_model_size * 4 - 3
                d3 = 70e9 / test_model_size * 4 - 3
                # Compute absolute difference for samples
                filtered = filtered.copy()  # avoid SettingWithCopyWarning
                filtered['code_diff'] = np.abs(filtered['coding_sampling'] - coding_sampling) / d1
                filtered['review_diff'] = np.abs(filtered['review_sampling'] - review_sampling) / d2
                filtered['test_diff'] = np.abs(filtered['test_sampling'] - test_sampling) / d3
                filtered['total_diff'] = filtered['code_diff'] + filtered['review_diff'] + filtered['test_diff']

                # Take the row with the smallest total difference
                min_diff = filtered['total_diff'].min()
                items = filtered[filtered['total_diff'] == min_diff]
        return items, min_diff

