import itertools
from typing import Dict

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

from rouge_score import rouge_scorer
from bert_score import score
from sklearn.metrics import precision_recall_fscore_support as prfs
import Levenshtein

from archon.completions.utils import custom_generator

from utils import precision_recall_f1, accuracy
from evaluation_2wiki import metric_evaluation

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

class ContentFormatError(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_prompt(dependency_type):
    if dependency_type == "resource":
        prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
    else:
        prompt = """\n# GOAL #:\nBased on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""
    return prompt

def check_format(content, dependency_type, model_name) -> Dict:
    content = content.replace("\n", "")
    content = content.replace("\_", "_")
    content = content.replace("\\", "")

    start_pos = content.find("RESULT #:")
    if start_pos != -1:
        content = content[start_pos + len("RESULT #:"):]
    content = content[content.find("{"):content.rfind("}") + 1]
    try:
        content_ = json.loads(content)
        if isinstance(content_, list) and len(content_):
            merge_content = {}
            for c in content_:
                for k, v in c.items():
                    merge_content[k].extend(v) if k in merge_content else merge_content.update({k: v})
        output = content_
        return output
    except json.JSONDecodeError as e:
        if dependency_type == "resource":
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps and task nodes;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. You must output the result in this schema: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
        else:
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
        prompt = prompt.replace("{{illegal_result}}", content)

        print(f"{('### warning:')} Illegal JSON format: {content}")

        messages = (
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                ]  # system
                + [{"role": "user", "content": prompt}]  # prompt
        )
        content = custom_generator(
            model_name, messages, max_tokens=1024, temperature=0.1, repeating=1
        )[0]
        content = content.replace("\n", "")
        content = content.replace("\_", "_")
        start_pos = content.find("STRICT JSON FORMAT #:")
        if start_pos != -1:
            content = content[start_pos + len("STRICT JSON FORMAT #:"):]
        content = content[content.find("{"):content.rfind("}") + 1]

        try:
            content = json.loads(content)
            return content
        except json.JSONDecodeError as e:
            raise ContentFormatError(f"{content}")



class Decompose(Component):

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
        print(f"Decompose initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):

        candidates = prev_state["candidates"]
        task_decomposition_results = self.decompose(conversation, candidates)
        state["candidates"].extend(task_decomposition_results)

        return

    def decompose(self, conversation: list, candidates: list):

        # conversation is the initial start
        user_request = conversation[-1]["content"]
        dependency_type = conversation[-1]["dependency_type"]
        demos = conversation[-1]["demos"]
        tool_string = conversation[-1]["tool_string"]

        prompt = get_prompt(dependency_type)

        if len(demos) > 0:
            prompt += "\n"
            for demo in demos:
                prompt += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""

        prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""

        prompt = tool_string + prompt.replace("{{user_request}}", user_request)

        if print_log:
            print('\n### Log, Decompose prompt', prompt)
            print("\n### Log, decompose, len(prompt.split())", len(prompt.split()))

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
        ) # output is a list with x_samples
        if print_log: print("\n### Log Decompose output[0]", output[0])
            # generations.append(content)

        if output is not None:
            generations.extend(output)
        if print_log:
            print("\n### Log Decompose output[1].split()", len(output[1].split()))

        return generations


class Merge(Component):
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
        print(f"Merge initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):
        candidates = prev_state["candidates"]
        tool_string = conversation[-1]["tool_string"]

        query = conversation[-1]["user_request"]
        prompt = f"You have been provided with a set of responses with the json format from various open-source models to the user request.\
                    Your task is to synthesize these responses into a single, high-quality response. \
                    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \
                    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \
                    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
        prompt += f"The user request is: {query}\n"

        prompt += "Task list: " + tool_string

        prompt += "Responses from models:"

        for i, reference in enumerate(candidates):
            json_content = check_format(reference, dependency_type, "meta-llama/Llama-3.1-70B-Instruct")
            s = json.dumps(json_content)
            prompt += f"\n\n{i + 1}. {s}"

        prompt += """\n\nThe format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""

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
        if print_log: print("\n### Log Merge output[0]", output[0])
        generations.extend(output)
        state["candidates"].extend(generations)

        return



class ApiSelection(Component):

    def __init__(self, config, custom_components=None):
        self.config = config

        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.samples = self.config["samples"]
        self.custom_components = custom_components

        self.model = Generator(self.config, custom_generators=custom_components)
        print(f"Api Selection initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):
        candidates = prev_state["candidates"]

        user_request = conversation[-1]["content"]
        dependency_type = conversation[-1]["dependency_type"]
        demos = conversation[-1]["demos"]
        tool_string = conversation[-1]["tool_string"]

        content = candidates[0]
        if print_log: print('\n### Log, ApiSelection content', content)
        json_content = check_format(content, dependency_type, "meta-llama/Llama-3.1-70B-Instruct")
        if print_log:
            print('\nLog, candidates to api selection:', json_content)
        #
        conversation[-1]['decompose_result'] = json_content
        #
        json_content["task_nodes"] = {}
        content_input = json.dumps(json_content)
        prompt = get_prompt(dependency_type)

        if len(demos) > 0:
            prompt += "\n"
            for demo in demos:
                prompt += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""

        prompt += """\n\n# USER REQUEST #: {{user_request}}\nYou have partial result with missing task nodes #: {{partial_content}}\nnow please generate your complete result in a strict JSON format:\n# RESULT #:"""
        prompt = tool_string + prompt.replace("{{user_request}}", user_request).replace("{{partial_content}}", content_input)
        if print_log:
            print("\n### Log, ApiSelection prompt.split()", len(prompt.split()))
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
        ) # output is a list with x_samples
        generations.extend(output)
        if print_log:
            print("\n### Log, ApiSelection output[0]", len(output[0].split()))

        state["candidates"].extend(generations)

        return

class ParameterSelection(Component):

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
        print(f"Parameter Selection initialized with model: {self.model_name}")

    def run(self, conversation: list, prev_state, state):
        candidates = prev_state["candidates"]
        generations = self.parameter_selection(conversation, candidates)
        state["candidates"].extend(generations)

        return

    def parameter_selection(self, conversation: list, candidates: list):
        # conversation is the initial start

        user_request = conversation[-1]["content"]
        dependency_type = conversation[-1]["dependency_type"]
        demos = conversation[-1]["demos"]
        tool_string = conversation[-1]["tool_string"]

        content = candidates[0]

        json_content = check_format(content, dependency_type, "meta-llama/Llama-3.1-70B-Instruct")
        if print_log:
            print('\n### Log, candidates to parameter selection:', json_content)
        #
        conversation[-1]['api_selection_result'] = json_content
        #
        task_nodes = json_content["task_nodes"]
        for task_node in task_nodes:
            task_node['arguments'] = []
        json_content["task_nodes"] = task_nodes

        content_input = json.dumps(json_content)
        prompt = get_prompt(dependency_type)

        if len(demos) > 0:
            prompt += "\n"
            for demo in demos:
                prompt += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""

        prompt += """\n\n# USER REQUEST #: {{user_request}}\nYou have partial result with missing arguments in task nodes #: {{partial_content}}\nnow please generate your complete result in a strict JSON format:\n# RESULT #:"""
        prompt = tool_string + prompt.replace("{{user_request}}", user_request).replace("{{partial_content}}",
                                                                                      content_input)
        if print_log:
            print("\n### Log, Parameter Selection prompt.split()", len(prompt.split()))
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
        )  # output is a list with x_samples
        generations.extend(output)
        if print_log:
            print("\n### Log, Parameter Selection output[0]", len(output[0]))

        return generations


def flatten(gt, pred, types = None):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()

        union.update(sample_gt)
        union.update(sample_pred)

        for s in union:
            if types:
                if s in types:
                    if s in sample_gt:
                        gt_flat.append(types.index(s)+1)
                    else:
                        gt_flat.append(0)

                    if s in sample_pred:
                        pred_flat.append(types.index(s)+1)
                    else:
                        pred_flat.append(0)
                else:
                    gt_flat.append(0)
                    pred_flat.append(0)
            else:
                if s in sample_gt:
                    gt_flat.append(1)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    pred_flat.append(1)
                else:
                    pred_flat.append(0)
    return gt_flat, pred_flat


def ratio_levenshtein(x, y):
    assert len(x) == len(y)
    n = len(x)
    total = 0
    for i in range(n):
        total += Levenshtein.ratio(x[i], y[i])
    return total / n


def get_content_type(content):
    content = content.strip('\'')
    assert isinstance(content, str), content
    # image
    for ext in ["jpg", "png", "jpeg", "gif", "bmp", "tiff", "svg", "ico"]:
        if "."+ext in content:
            return "image"
    # audio
    for ext in ["mp3", "wav", "wma", "ogg", "aac", "flac", "aiff", "au"]:
        if "."+ext in content:
            return "audio"
    # video
    for ext in ["mp4", "avi", "mov", "flv", "wmv", "mkv", "webm", "m4v", "mpg", "mpeg"]:
        if "."+ext in content:
            return "video"
    return "text"

def evaluate(id, decompose_result, api_selection_result, parameter_selection_result, labels, s, n, metric, tool_desc, tool_map, tool_output_type_map, tool_map_reverse,dependency_type, alignment):
    label = labels[id]
    if print_log: print('### Log, label', label)
    metric_dict = {}


    # decompose
    prediction_task_step = "\n".join(decompose_result["task_steps"])
    label_task_step = "\n".join(label["tool_steps"])
    # rouge
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction_task_step, label_task_step)
    rougeL = scores["rougeL"]
    rouge1 = scores["rouge1"]
    rouge2 = scores["rouge2"]
    metric_dict[f"step_rouge1"] = rouge1.fmeasure
    metric_dict[f"step_rouge2"] = rouge2.fmeasure
    metric_dict[f"step_rougeL"] = rougeL.fmeasure

    # bertscore
    bert_P, bert_R, bert_F1 = score([prediction_task_step], [label_task_step], lang="en", model_type="bert-base-uncased", verbose=True)

    metric_dict[f"step_bertscore_precision"] = bert_P.item()
    metric_dict[f"step_bertscore_recall"] = bert_R.item()
    metric_dict[f"step_bertscore_f1"] = bert_F1.item()


    # tool selection
    label_nodes = label['tool_nodes']
    prediction_nodes = api_selection_result['task_nodes']
    label_node_name = [node['task'] for node in label_nodes]
    prediction_node_name = [node['task'] for node in prediction_nodes]
    if dependency_type == "resource":
        prediction_node_name = [name.replace("_", " ") for name in prediction_node_name]
        label_node_name = [name.replace("_", " ") for name in label_node_name]

    # if 'f1' in metric:
    types = list(range(1, len(tool_desc["nodes"])+1))
    types_name = [tool_map_reverse[i] for i in types]
    gt_flat, pred_flat = flatten([label_node_name], [prediction_node_name], types=types_name)
    if print_log: print('### Log label_node_name, prediction_node_name, types_name', label_node_name, prediction_node_name, types_name)
    micro = prfs(gt_flat, pred_flat, labels=types, average='micro')[:-1]
    macro = prfs(gt_flat, pred_flat, labels=types, average='macro')[:-1]
    metric_dict["node_micro_precision_no_matching"] = micro[0]
    metric_dict["node_micro_recall_no_matching"] = micro[1]
    metric_dict["node_micro_f1_no_matching"] = micro[2]
    metric_dict["node_macro_precision_no_matching"] = macro[0]
    metric_dict["node_macro_recall_no_matching"] = macro[1]
    metric_dict["node_macro_f1_no_matching"] = macro[2]
    num_intersection = len(set(label_node_name) & set(prediction_node_name))
    p = num_intersection / len(prediction_node_name)
    r = num_intersection / len(label_node_name)
    if p+r <=0.:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    metric_dict['node_f1'] = f1
    metric_dict['node_precision'] = p
    metric_dict['node_recall'] = r

    # ed
    # if "ed" in metric:
    labels = []
    predictions = []
    labels.append([tool_map.get(name, 0) for name in label_node_name])
    predictions.append([tool_map.get(name, 0) for name in prediction_node_name])
    if print_log: print('### Log, ED, labels, prediction', labels, predictions)
    ed = ratio_levenshtein(predictions, labels)
    metric_dict["edit_distance"] = 1-ed


    # parameter selection
    prediction_nodes = parameter_selection_result['task_nodes']
    prediction_node_name = [node['task'] for node in prediction_nodes]
    label_nodes = label['tool_nodes']
    label_node_name = [node["task"] for node in label_nodes]
    prediction_node_argument = [node.get("arguments", []) for node in prediction_nodes]
    label_node_argument = [node["arguments"] for node in label_nodes]

    if dependency_type == "resource":
        prediction_node_name = [name.replace("_", " ") for name in prediction_node_name]
        label_node_name = [name.replace("_", " ") for name in label_node_name]
        label_link = []
        prediction_link = []
        for inx, node in enumerate(label_nodes):
            new_arguments = []
            for i, argument in enumerate(node["arguments"]):
                try:
                    if isinstance(argument, dict):
                        argument = list(argument.values())[0]

                    if isinstance(argument, list):
                        argument = " ".join(argument)

                    if "<node-" in argument:
                        index_start = argument.index("<node-") + 6
                        index_end = argument.index(">")
                        if int(argument[index_start: index_end]) == inx+1:
                            continue
                        argument_tool_name = label_node_name[int(argument[index_start: index_end])]
                        label_link.append({"source": argument_tool_name, "target": node["task"]})
                        new_argument = {"name": tool_output_type_map.get(argument_tool_name, "other"), "value": argument_tool_name}
                    else:
                        new_argument = {"name": get_content_type(argument), "value": argument}
                except Exception as e:
                    pass
                new_arguments.append(new_argument)
            node["arguments"] = new_arguments
        for inx, node in enumerate(prediction_nodes):
            new_arguments = []
            for i, argument in enumerate(node.get("arguments", [])):
                try:
                    if isinstance(argument, dict):
                        argument = list(argument.values())[0]
                    if isinstance(argument, list):
                        argument = " ".join(argument)
                    if isinstance(argument, str) and "<node-" in argument:
                        index_start = argument.index("<node-") + 6
                        index_end = argument.index(">")

                        if int(argument[index_start: index_end]) == inx:
                            continue
                        prediction_tool_name = prediction_node_name[int(argument[index_start: index_end])]
                        prediction_link.append({"source": prediction_tool_name, "target": node["task"]})
                        new_argument = {"name": tool_output_type_map.get(prediction_tool_name, "other"), "value": prediction_tool_name}
                    else:
                        new_argument = {"name": get_content_type(argument), "value": argument}

                except Exception as e:
                    pass
                new_arguments.append(new_argument)
            node["arguments"] = new_arguments

    label_task_arg_name = []
    label_task_arg_name_value = []
    for task, arguments in zip(label_node_name, label_node_argument):
        for argument in arguments:
            if "<node-" in argument:
                index_start = argument.index("<node-") + 6
                index_end = argument.index(">")
                ind = int(argument[index_start: index_end])-1
                argument = f"<node-{ind}>"
            label_task_arg_name.append(f"{task}-{argument}")
            # label_task_arg_name.append(f"{task}-{argument['name']}")
            # label_task_arg_name_value.append(f"{task}-{argument['name']}-{argument['value']}")
    prediction_task_arg_name=[]
    prediction_task_arg_name_value=[]
    for task, arguments in zip (prediction_node_name, prediction_node_argument):
        for argument in arguments:
            prediction_task_arg_name.append(f"{task}-{argument}")
            # prediction_task_arg_name.append(f"{task}-{argument['name']}")
            # prediction_task_arg_name_value.append(f"{task}-{argument['name']}-{argument['value']}")

    gt_flat, pred_flat = flatten([label_task_arg_name], [prediction_task_arg_name])
    micro = prfs(gt_flat, pred_flat, average="binary")[:-1]
    if print_log:
        print("### Log prediction_task_arg_name:", prediction_task_arg_name)
        print("### Log label task arg name:", label_task_arg_name)
    if print_log: print(f"Argument Task-ArgName Binary F1: [ No Matching ]: {micro[-1]}")
    num_intersection = len(set(label_task_arg_name) & set(prediction_task_arg_name))
    r = num_intersection / len(label_task_arg_name)
    p = num_intersection / len(prediction_task_arg_name)
    if p + r <=0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    metric_dict["argument_task_argname_binary_f1_no_matching"] = micro[-1]
    metric_dict["argument_f1"] = f1
    metric_dict["argument_precision"] = p
    metric_dict["argument_recall"] = r

    # gt_flat, pred_flat = flatten([label_task_arg_name_value], [prediction_task_arg_name_value])
    # micro = prfs(gt_flat, pred_flat, average="binary")[:-1]
    # print(f"Argument Task-ArgName-Value Binary F1 [ No Matching ]: {micro[-1]}")
    # metric_dict["argument_task_argname_value_binary_f1_no_matching"] = micro[-1]


    return metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default="", help="Do not retrieve references.")
    parser.add_argument('--decomposer_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to retrieve.')
    parser.add_argument('--api_selection_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to generate.')
    parser.add_argument('--parameter_selection_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help='Model name to generate.')
    parser.add_argument('--decompose_samples', type=float, default=1., help='number of retriever sampling')
    parser.add_argument('--api_selection_samples', type=float, default=1., help='number of generation sampling')
    parser.add_argument('--parameter_selection_samples', type=float, default=1., help='number of generation sampling')
    parser.add_argument('--examples', type=int, default=100,)
    parser.add_argument('--test', action='store_true', help="test or train.")

    args = parser.parse_args()
    args.decompose_samples = int(args.decompose_samples)
    args.api_selection_samples = int(args.api_selection_samples)
    args.parameter_selection_samples = int(args.parameter_selection_samples)
    if args.test:
        args.examples = 500

    wandb.init(project="taskbench", config=args)
    args = wandb.config

    decomposer_model_name = args.decomposer_model_name
    api_selection_model_name = args.api_selection_model_name
    parameter_selection_model_name = args.parameter_selection_model_name

    decompose_samples = args.decompose_samples
    api_selection_samples = args.api_selection_samples
    parameter_selection_samples = args.parameter_selection_samples

    # if decomposer_model_name == parameter_selection_model_name:
    #     wandb.finish()
    #     print('Two models cannot be the same.')
    #     sys.exit(0)
    # if decomposer_model_name == api_selection_model_name:
    #     wandb.finish()
    #     print('Two models cannot be the same.')
    #     sys.exit(0)
    decompose_model_size = model_size_mapping[decomposer_model_name]
    api_selection_model_size = model_size_mapping[api_selection_model_name]
    parameter_selection_model_size = model_size_mapping[parameter_selection_model_name]

    if decompose_model_size>=70:
        if decompose_samples > 10:
            wandb.finish()
            print('budget is not enough!!!')
            sys.exit(0)

    if api_selection_model_size>=70:
        if api_selection_samples > 10:
            wandb.finish()
            print('budget is not enough!!!')
            sys.exit(0)

    if parameter_selection_model_size>=70:
        if parameter_selection_samples > 10:
            wandb.finish()
            print('budget is not enough!!!')
            sys.exit(0)

    task_name = args.task_name
    num_examples = args.examples
    print("task_name:", task_name)
    print("decomposer model name:", decomposer_model_name)
    print("api selection model name:", api_selection_model_name)
    print("parameter selection model name:", parameter_selection_model_name)
    print("number of decompose samples:", decompose_samples)
    print("number of api selection samples:", api_selection_samples)
    print("number of parameter_selection_samples:", parameter_selection_samples)

    # retrieve-merge-generate
    archon_config = {
        "name": "archon-taskbench",
        "custom": True,  # SET THE CONFIG TO USE CUSTOM
        "layers": [
            [
                {
                    "type": "decompose",  # custom type here
                    "model": decomposer_model_name,
                    "model_type": "custom_qwen",  # which api or which Generator
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "samples": decompose_samples,
                },
            ],
            [
                {
                    "type": "merge",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "model_type": "custom_qwen",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "samples": 1
                }
            ],
            [
                {
                    "type": "api_selection",
                    "model": api_selection_model_name,
                    "model_type": "custom_qwen",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "samples": api_selection_samples,
                }
            ],
            [
                {
                    "type": "merge",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "model_type": "custom_qwen",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "samples": 1
                }
            ],
            [
                {
                    "type": "parameter_selection",
                    "model": parameter_selection_model_name,
                    "model_type": "custom_qwen",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "samples": parameter_selection_samples,
                }
            ],
            [
                {
                    "type": "merge",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "model_type": "custom_qwen",
                    "temperature": 0.3,
                    "max_tokens": 2048,
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
    archon.add_component("decompose", Decompose)
    archon.add_component("api_selection", ApiSelection)
    archon.add_component("parameter_selection", ParameterSelection)
    archon.add_component("merge", Merge)
    archon.add_generator("custom_qwen", custom_generator)

    # Have to manually initialize
    archon.initialize()

    # load data
    data_dir = f"data/taskbench/{task_name}"
    if task_name == "data_dailylifeapis":
        dependency_type = "temporal"
    else:
        dependency_type = "resource"
    tool_desc = json.load(open(f"data/taskbench/{task_name}/tool_desc.json"))['nodes']
    if dependency_type == "temporal":
        for tool in tool_desc:
            parameter_list = []
            for parameter in tool["parameters"]:
                parameter_list.append(parameter["name"])
            tool["parameters"] = parameter_list
    demos = []
    if dependency_type == "temporal":
        demos_id = ["38563456", "27267145", "91005535"]
    else:
        if "huggingface" in data_dir:
            demos_id = ["10523150", "14611002", "22067492"]
        elif "multimedia" in data_dir:
            demos_id = ["30934207", "20566230", "19003517"]
    demos_id = demos_id[:1]

    demos_rf = open(f"data/taskbench/{task_name}/data.json", "r")
    for line in demos_rf:
        data = json.loads(line)
        if data["id"] in demos_id:
            if dependency_type == "temporal":
                demo = {
                    "user_request": data["instruction"],
                    "result": {
                        "task_steps": data["tool_steps"],
                        "task_nodes": data["tool_nodes"],
                        "task_links": data["tool_links"]
                    }
                }
            else:
                demo = {
                    "user_request": data["instruction"],
                    "result": {
                        "task_steps": data["tool_steps"],
                        "task_nodes": data["tool_nodes"]
                    }
                }
            demos.append(demo)
    demos_rf.close()

    tool_string = "# TASK LIST #:\n"
    for k, tool in enumerate(tool_desc):
        tool_string += json.dumps(tool) + "\n"

    rf_ur = open(f"data/taskbench/{task_name}/user_requests.json", "r")
    examples = []
    count = 0
    if args.test:
        for line in rf_ur:
            example = json.loads(line)
            examples.append(example)
        examples = examples[100: 100 + num_examples]
    else:
        for line in rf_ur:
            example = json.loads(line)
            examples.append(example)
            count += 1
            if count >= num_examples:
                break
    rf_ur.close()

    # for evaluation
    tool_desc = json.load(open(f"data/taskbench/{task_name}/tool_desc.json", "r"))
    tool_map = {tool["id"]: i+1 for i, tool in enumerate(tool_desc["nodes"])}
    tool_map_reverse = {i+1: tool["id"] for i, tool in enumerate(tool_desc["nodes"])}
    tool_map_reverse[0] = "NEGATIVE"
    tool_map["<PAD>"] = -1

    tool_output_type_map = None
    if dependency_type == "resource":
        tool_output_type_map = {tool["id"]: tool["output-type"][0] if len(tool["output-type"]) else "none" for tool in tool_desc["nodes"]}

    splits = ['all']
    n_tools = ['all']

    if "all" in splits:
        splits = ["overall", "single", "chain", "dag", ]
    if "all" in n_tools:
        n_tools = ["overall"] + [str(i) for i in range(1, 11)]

    alignment = ""  # all
    alignment_ids = None
    # with open(f"{data_dir}/alignment_ids.json", "r") as alignment_file:
    #     alignment_ids = json.load(alignment_file)
    #     alignment_ids = list(itertools.chain(*alignment_ids[f"{alignment}_alignment_id"].values()))
    #     print(f"Alignment Mode: {alignment} ({len(alignment_ids)})")

    mode = "mul"
    group = []
    if mode == "mul":
        for s in splits:
            for n in n_tools:
                if (s, n) not in group:
                    group.append((s, n))
    elif mode == "add":
        for s in splits:
            if (s, "overall") not in group:
                group.append((s, "overall"))
        for n in n_tools:
            if ("overall", n) not in group:
                group.append(("overall", n))
    else:
        assert False, "mode should be mul or add"

    labels_all = {}
    labels = {}
    for s, n in group:
        labels_ = {}
        with open(f"data/taskbench/{task_name}/data.json", "r") as label_rf:
            for line in label_rf:
                line = line.replace('\\"', '"').replace('"[', '[').replace(']"', ']')
                try:
                    data = json.loads(line)
                except:
                    continue
                real_tool_num = len(data["tool_nodes"])
                if alignment_ids is None or data["id"] in alignment_ids:
                    if s == "overall" or data["type"] == s:
                        if n == "overall" or str(real_tool_num) == n:
                            id = data["id"]
                            labels_[id] = data
                            labels[id] = data
        labels_all[(s,n)] = labels_

    metric_all = {}
    unsuccess_num = 0
    for example in tqdm.tqdm(examples):
        if print_log:
            print('log user request,', example['user_request'])

        # make conversation
        example['role'] = 'user'
        example['content'] = example['user_request']
        example['tool_string'] = tool_string
        example['demos'] = demos
        example['dependency_type'] = dependency_type
        testing_instruction = [example]
        try:
            response = archon.generate(testing_instruction)
            content = response
            json_content = check_format(content, dependency_type, "meta-llama/Llama-3.1-70B-Instruct")
        except:
            print('error........................')
            unsuccess_num += 1
            continue
        if print_log: print('\n\n### Log final output json content:', json_content)
        # 三个模型的输出
        decompose_result = example['decompose_result']
        api_selection_result = example['api_selection_result']
        parameter_selection_result = json_content
        if print_log: print('\n### type of results:', type(decompose_result), type(api_selection_result), type(parameter_selection_result))

        # evaluate
        metric = ["f1", "ed", "link", "argument", "rouge", "bertscore"]
        id = example['id']
        # for s, n in group:
        #     labels = labels_all[(s,n)]
        try:
            metric_dict = evaluate(id, decompose_result, api_selection_result, parameter_selection_result, labels, 0, 0, metric, tool_desc, tool_map, tool_output_type_map, tool_map_reverse,dependency_type, alignment)
        except:
            metric_dict = {}
            print('error during evaluation........................')
        if print_log: print('\n### Log metrics:', metric_dict)
        for key, value in metric_dict.items():
            if key not in metric_all:
                metric_all[key] = [value]
            else:
                metric_all[key].append(value)
    result_dict = {}
    for k, v in metric_all.items():
        result_dict[k] = np.mean(v)
    result_dict['unsuccess_num'] = unsuccess_num
    wandb.log(result_dict)
    wandb.finish()

'''
python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 2 --api_selection_model_name meta-llama/Llama-3.1-8B-Instruct --api_selection_samples 2 --parameter_selection_model_name meta-llama/Llama-3.1-8B-Instruct --parameter_selection_samples 2 --examples 1

test scripts for dailyapius
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 20 --api_selection_model_name meta-llama/Llama-3.1-70B-Instruct --api_selection_samples 3 --parameter_selection_model_name meta-llama/Llama-3.1-70B-Instruct --parameter_selection_samples 1 --test > log1.log 2>&1 &
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 1 --api_selection_model_name meta-llama/Llama-3.1-8B-Instruct --api_selection_samples 50 --parameter_selection_model_name meta-llama/Llama-3.1-70B-Instruct --parameter_selection_samples 1 --test > log2.log 2>&1 &
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-70B-Instruct --decompose_samples 4 --api_selection_model_name meta-llama/Llama-3.1-8B-Instruct --api_selection_samples 20 --parameter_selection_model_name meta-llama/Llama-3.1-70B-Instruct --parameter_selection_samples 1 --test > log3.log 2>&1 &
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 1 --api_selection_model_name meta-llama/Llama-3.1-8B-Instruct --api_selection_samples 50 --parameter_selection_model_name meta-llama/Llama-3.1-8B-Instruct --parameter_selection_samples 1 --test > log4.log 2>&1 &
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 1 --api_selection_model_name meta-llama/Llama-3.1-70B-Instruct --api_selection_samples 2 --parameter_selection_model_name meta-llama/Llama-3.1-70B-Instruct --parameter_selection_samples 1 --test > log5.log 2>&1 &
nohup python archon_taskbench.py --task_name data_multimedia --decomposer_model_name meta-llama/Llama-3.1-8B-Instruct --decompose_samples 10 --api_selection_model_name meta-llama/Llama-3.1-8B-Instruct --api_selection_samples 1 --parameter_selection_model_name meta-llama/Llama-3.1-8B-Instruct --parameter_selection_samples 1 --test > log6.log 2>&1 &

# best config
# ours: decomp, meta-llama/Llama-3.1-8B-Instruct 20, meta-llama/Llama-3.1-70B-Instruct 3, meta-llama/Llama-3.1-70B-Instruct 1 
# hpo: decomp, meta-llama/Llama-3.1-8B-Instruct 1, meta-llama/Llama-3.1-8B-Instruct 50, meta-llama/Llama-3.1-70B-Instruct 1
# mlcpoilot: decomp, meta-llama/Llama-3.1-70B-Instruct 4, meta-llama/Llama-3.1-8B-Instruct 20, meta-llama/Llama-3.1-70B-Instruct 1 0.527
# zs: decomp, meta-llama/Llama-3.1-8B-Instruct 1, meta-llama/Llama-3.1-8B-Instruct 50, meta-llama/Llama-3.1-8B-Instruct 1
# bayes: decomp, meta-llama/Llama-3.1-8B-Instruct 1, meta-llama/Llama-3.1-70B-Instruct 2, meta-llama/Llama-3.1-70B-Instruct 1 0.52
# random: decomp, meta-llama/Llama-3.1-8B-Instruct 10, meta-llama/Llama-3.1-8B-Instruct 1, meta-llama/Llama-3.1-8B-Instruct 1 0.431

'''

