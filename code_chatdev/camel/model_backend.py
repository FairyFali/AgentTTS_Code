# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict
from collections import defaultdict

import openai
import tiktoken
import time
from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize

try:
    from openai.types.chat import ChatCompletion

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version

import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
if 'BASE_URL' in os.environ:
    BASE_URL = os.environ['BASE_URL']
else:
    BASE_URL = None

print_log = False

class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""
    instance_count = defaultdict(int)
    method_calls_count = defaultdict(int)

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


    @classmethod
    def get_stats(cls):
        return {
            'instance_count': dict(cls.instance_count),
            'method_calls_count': dict(cls.method_calls_count)
        }


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, repeating, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.repeating = repeating
        self.model_config_dict = model_config_dict
        OpenAIModel.instance_count[self.model_type.value] += 1

    def run(self, *args, **kwargs):

        OpenAIModel.method_calls_count[self.model_type.value] += 1

        string = "\n".join([message["content"] for message in kwargs["messages"]])
        value_for_tiktoken = self.model_type.value
        if 'llama' in value_for_tiktoken or 'qwen' in value_for_tiktoken or 'deepseek-chat' in value_for_tiktoken:
            encoding = tiktoken.get_encoding('cl100k_base')
        else:
            encoding = tiktoken.encoding_for_model(self.model_type.value)
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        if openai_new_api:
            # print('#### Using new open ai')
            # Experimental, add base_url
            if BASE_URL:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=BASE_URL,
                )
            else:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY
                )

            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-16k-0613": 16384,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 100000,
                "gpt-4o": 4096, #100000
                "gpt-4o-mini": 16384, #100000
                "llama3.2:1b": 8192,
                "llama3.2": 8192,
                "llama3.2:3b": 8192,
                "llama3.1:8b": 8192,
                "llama3.1:70b": 8192,
                "llama3.1:405b": 8192,
                "qwen2.5:3b": 8192,
                "qwen2.5:7b": 8192,
                "qwen2.5:14b": 8192,
                "qwen2.5:32b": 8192,
                "qwen2.5:72b": 8192,

            }
            num_max_token = num_max_token_map[self.model_type.value]
            num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = num_max_completion_tokens
            if self.repeating <= 1:
                # model type value: LLAMA_3B = "llama3.2:3b"
                response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                          **self.model_config_dict)

            else:
                responses = []
                # 获得多个responses
                for _ in range(self.repeating):
                    # print('###repeating', _)
                    response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                          **self.model_config_dict)
                    time.sleep(1)
                    responses.append(response)
                # 合并这些responses
                references = [response.choices[0].message.content for response in responses]
                query = kwargs.get("messages")[-1]["content"]
                prompt = f"You have been provided with a set of responses from various open-source models to the latest user query, which is {query}.\
                    Your task is to synthesize these responses into a single, high-quality response. \
                    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \
                    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \
                    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
                # prompt += f"Once again, the query is: {query}\n"

                # prompt += "Responses from models:"

                for i, reference in enumerate(references):
                    prompt += f"\n{i + 1}. {reference}"

                kwargs["messages"] = [{'role':"system", "content": "You are a helpful assistant who fuses multiple answers."}, {'role':"user", "content": prompt}]
                # print('Log, OpenAIModel run merging prompt:', prompt, 'Ending log #############')
                response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                          **self.model_config_dict)
                # print('Log, OpenAIModel run merging response:', response)

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response.usage.prompt_tokens,
                num_completion_tokens=response.usage.completion_tokens
            )
            if print_log:
                log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                    response.usage.total_tokens, cost))
            if not isinstance(response, ChatCompletion):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
        else:
            # print('#### Using old open ai')
            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-16k-0613": 16384,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 100000,
                "gpt-4o": 4096, #100000
                "gpt-4o-mini": 16384, #100000
                "llama3.2:1b": 8192,
                "llama3.2": 8192,
                "llama3.2:3b": 8192,
                "llama3.1:8b": 8192,
                "llama3.1:70b": 8192,
                "llama3.1:405b": 8192,
                "qwen2.5:3b": 8192,
                "qwen2.5:7b": 8192,
                "qwen2.5:14b": 8192,
                "qwen2.5:32b": 8192,
                "qwen2.5:72b": 8192,
            }
            num_max_token = num_max_token_map[self.model_type.value]
            num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = openai.ChatCompletion.create(*args, **kwargs, model=self.model_type.value,
                                                    **self.model_config_dict)

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response["usage"]["prompt_tokens"],
                num_completion_tokens=response["usage"]["completion_tokens"]
            )

            if print_log: log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                    response["usage"]["total_tokens"], cost))
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, repeating, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type in {
            ModelType.GPT_3_5_TURBO, ModelType.GPT_3_5_TURBO.value,
            ModelType.GPT_3_5_TURBO_NEW, ModelType.GPT_3_5_TURBO_NEW.value,
            ModelType.GPT_4, ModelType.GPT_4.value,
            ModelType.GPT_4_32k, ModelType.GPT_4_32k.value,
            ModelType.GPT_4_TURBO, ModelType.GPT_4_TURBO.value,
            ModelType.GPT_4_TURBO_V, ModelType.GPT_4_TURBO_V.value,
            ModelType.GPT_4O, ModelType.GPT_4O.value,
            ModelType.GPT_4O_MINI, ModelType.GPT_4O_MINI.value,
            ModelType.LLAMA_1B, ModelType.LLAMA_1B.value,
            ModelType.LLAMA_3B, ModelType.LLAMA_3B.value,
            ModelType.LLAMA_8B, ModelType.LLAMA_8B.value,
            ModelType.LLAMA_70B, ModelType.LLAMA_70B.value,
            ModelType.QWEN_3B, ModelType.QWEN_3B.value,
            ModelType.QWEN_7B, ModelType.QWEN_7B.value,
            ModelType.QWEN_14B, ModelType.QWEN_14B.value,
            ModelType.QWEN_32B, ModelType.QWEN_32B.value,
            ModelType.QWEN_72B, ModelType.QWEN_72B.value,
            None
        }:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, repeating, model_config_dict)
        return inst
