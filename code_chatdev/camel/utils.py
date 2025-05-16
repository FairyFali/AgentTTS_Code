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
import os
import re
import zipfile
from functools import wraps
from typing import Any, Callable, List, Optional, Set, TypeVar

import requests
import tiktoken

from camel.messages import OpenAIMessage
from camel.typing import ModelType, TaskType

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, BitsAndBytesConfig
from transformers import MllamaForConditionalGeneration
import torch.nn.functional as F
import torch

F = TypeVar('F', bound=Callable[..., Any])

import time


def count_tokens_openai_chat_models(
        messages: List[OpenAIMessage],
        encoding: Any,
) -> int:
    r"""Counts the number of tokens required to generate an OpenAI chat based
    on a given list of messages.

    Args:
        messages (List[OpenAIMessage]): The list of messages.
        encoding (Any): The encoding method to use.

    Returns:
        int: The number of tokens required.
    """
    num_tokens = 0
    for message in messages:
        # message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_from_messages(
        messages: List[OpenAIMessage],
        model: ModelType,
) -> int:
    r"""Returns the number of tokens used by a list of messages.

    Args:
        messages (List[OpenAIMessage]): The list of messages to count the
            number of tokens for.
        model (ModelType): The OpenAI model used to encode the messages.

    Returns:
        int: The total number of tokens used by the messages.

    Raises:
        NotImplementedError: If the specified `model` is not implemented.

    References:
        - https://github.com/openai/openai-python/blob/main/chatml.md
        - https://platform.openai.com/docs/models/gpt-4
        - https://platform.openai.com/docs/models/gpt-3-5
    """
    try:
        value_for_tiktoken = model.value_for_tiktoken
        encoding = tiktoken.encoding_for_model(value_for_tiktoken)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        ModelType.GPT_3_5_TURBO,
        ModelType.GPT_3_5_TURBO_NEW,
        ModelType.GPT_4,
        ModelType.GPT_4_32k,
        ModelType.GPT_4_TURBO,
        ModelType.GPT_4_TURBO_V,
        ModelType.GPT_4O,
        ModelType.GPT_4O_MINI,
        ModelType.LLAMA_1B,
        ModelType.LLAMA_3B,
        ModelType.LLAMA_8B,
        ModelType.LLAMA_70B,
        ModelType.LLAMA_405B,
        ModelType.QWEN_3B,
        ModelType.QWEN_7B,
        ModelType.QWEN_14B,
        ModelType.QWEN_32B,
        ModelType.QWEN_72B,
        ModelType.STUB
    }:
        return count_tokens_openai_chat_models(messages, encoding)
    else:
        raise NotImplementedError(
            f"`num_tokens_from_messages`` is not presently implemented "
            f"for model {model}. "
            f"See https://github.com/openai/openai-python/blob/main/chatml.md "
            f"for information on how messages are converted to tokens. "
            f"See https://platform.openai.com/docs/models/gpt-4"
            f"or https://platform.openai.com/docs/models/gpt-3-5"
            f"for information about openai chat models.")


def get_model_token_limit(model: ModelType) -> int:
    r"""Returns the maximum token limit for a given model.

    Args:
        model (ModelType): The type of the model.

    Returns:
        int: The maximum token limit for the given model.
    """
    if model == ModelType.GPT_3_5_TURBO or model == ModelType.GPT_3_5_TURBO.value:
        return 16384
    elif model == ModelType.GPT_3_5_TURBO_NEW or model == ModelType.GPT_3_5_TURBO_NEW.value:
        return 16384
    elif model == ModelType.GPT_4 or model == ModelType.GPT_4.value:
        return 8192
    elif model == ModelType.GPT_4_32k or model == ModelType.GPT_4_32k.value:
        return 32768
    elif model == ModelType.GPT_4_TURBO or model == ModelType.GPT_4_TURBO.value:
        return 128000
    elif model == ModelType.STUB or model == ModelType.STUB.value:
        return 4096
    elif model == ModelType.GPT_4O or (model == ModelType.GPT_4O.value):
        return 128000
    elif model == ModelType.GPT_4O_MINI or (model == ModelType.GPT_4O_MINI.value):
        return 128000
    elif model in [ModelType.LLAMA_8B, ModelType.LLAMA_1B, ModelType.LLAMA_3B, ModelType.LLAMA_3B.value, ModelType.LLAMA_70B, ModelType.LLAMA_70B.value, ModelType.LLAMA_405B, ModelType.LLAMA_405B.value] or model == ModelType.LLAMA_8B.value or model == ModelType.LLAMA_1B.value:
        return 128000
    elif model in [ModelType.QWEN_3B, ModelType.QWEN_7B, ModelType.QWEN_14B, ModelType.QWEN_72B]:
        return 128000
    else:
        raise ValueError("Unknown model type")


def openai_api_key_required(func: F) -> F:
    r"""Decorator that checks if the OpenAI API key is available in the
    environment variables.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The decorated function.

    Raises:
        ValueError: If the OpenAI API key is not found in the environment
            variables.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from camel.agents.chat_agent import ChatAgent
        if not isinstance(self, ChatAgent):
            raise ValueError("Expected ChatAgent")
        if self.model == ModelType.STUB:
            return func(self, *args, **kwargs)
        elif 'OPENAI_API_KEY' in os.environ:
            return func(self, *args, **kwargs)
        else:
            raise ValueError('OpenAI API key not found.')

    return wrapper


def print_text_animated(text, delay: float = 0.005, end: str = ""):
    r"""Prints the given text with an animated effect.

    Args:
        text (str): The text to print.
        delay (float, optional): The delay between each character printed.
            (default: :obj:`0.02`)
        end (str, optional): The end character to print after the text.
            (default: :obj:`""`)
    """
    for char in text:
        print(char, end=end, flush=True)
        time.sleep(delay)
    print('\n')


def get_prompt_template_key_words(template: str) -> Set[str]:
    r"""Given a string template containing curly braces {}, return a set of
    the words inside the braces.

    Args:
        template (str): A string containing curly braces.

    Returns:
        List[str]: A list of the words inside the curly braces.

    Example:
        >>> get_prompt_template_key_words('Hi, {name}! How are you {status}?')
        {'name', 'status'}
    """
    return set(re.findall(r'{([^}]*)}', template))


def get_first_int(string: str) -> Optional[int]:
    r"""Returns the first integer number found in the given string.

    If no integer number is found, returns None.

    Args:
        string (str): The input string.

    Returns:
        int or None: The first integer number found in the string, or None if
            no integer number is found.
    """
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None


def download_tasks(task: TaskType, folder_path: str) -> None:
    # Define the path to save the zip file
    zip_file_path = os.path.join(folder_path, "tasks.zip")

    # Download the zip file from the Google Drive link
    response = requests.get("https://huggingface.co/datasets/camel-ai/"
                            f"metadata/resolve/main/{task.value}_tasks.zip")

    # Save the zip file
    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    # Delete the zip file
    os.remove(zip_file_path)






# add it
model_dict = {
    "": {
        'model': None,
        'tokenizer': None
    }
}

def load_qwen_model(model_name, device_map='auto'):
    '''
    load awq qwen.
    '''
    if 'awq' in model_name.lower():
        return load_qwen_awq_model(model_name, device_map=device_map)
    else:
        return load_model(model_name, device_map=device_map)


def load_qwen_awq_model(model_name, device_map='auto'):
    assert 'awq' in model_name.lower()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_model(model_name, device_map='auto', use_4bit=True):
    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant, )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map=device_map, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_llama_model(model_name, device_map='auto'):
    '''
    quantized llama
    '''
    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant, )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map=device_map, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_vision_llama_model(model_name, device_map='auto'):
    # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model_id = model_name
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_gemma_model(model_name, device_map="auto"):
    '''
    load gemma from (google/gemma-2-2b-it, google/gemma-2-9b-it, google/gemma-2-27b-it,) and quantize
    :param model_name:
    :param device_map:
    :return:
    '''

    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant, )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map=device_map, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_gemma3_model(model_name, device_map="auto"):
    '''
    load gemma from (google/gemma-3-4b-it,) and quantize
    transformer版本太老，不能加载gemma-3
    :param model_name:
    :param device_map:
    :return:
    '''

    # use_4bit = True
    # bnb_4bit_compute_dtype = 'float16'
    # bnb_4bit_quant_type = "nf4"
    # use_nested_quant = True
    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant, )
    # # Load model directly
    #
    # processor = AutoProcessor.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    # model = AutoModelForImageTextToText.from_pretrained(model_name)
    #
    # return model, processor
    pass

def load_phi_model(model_name, device_map="auto"):
    '''
    load phi from (microsoft/Phi-3-mini-4k-instruct 3.8B, microsoft/Phi-3-small-8k-instruct 7B, microsoft/Phi-3-medium-4k-instruct 14B )
    :param model_name:
    :param device_map:
    :return:
    '''

    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant, )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map=device_map, quantization_config=bnb_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer


def get_model_by_name(name):
    if name in model_dict:
        return model_dict[name]['model'], model_dict[name]['tokenizer']
    else:
        if 'Qwen' in name:
            model, tokenizer = load_qwen_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'llama' in name.lower() and 'vision' in name.lower():
            model, processor = load_vision_llama_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = processor
            return model, processor
        elif 'llama' in name:
            model, tokenizer = load_llama_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'gemma' in name:
            if "gemma-3" in name:
                model, tokenizer = load_gemma3_model(name)
            else:
                model, tokenizer = load_gemma_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'Phi' in name:
            model, tokenizer = load_phi_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        else:
            pass


def custom_generator(model, messages, image=None, max_tokens=1024, temperature=0.7, repeating=1, **kwargs):
    if 'gemma' in model:
        if messages[0]['role'] == 'system':
            messages = messages[1:]
    model, tokenizer = get_model_by_name(model)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, padding_side='left',
                             truncation=True).to(model.device)

    if repeating > 1:
        bs = 1
        responses_list = [[] for _ in range(bs)]  # no [[]]* bs, the inner [] is shared
        # 修改为每一个batch最大10，否则容易OOM
        for _r in range(0, repeating, 10):
            if repeating - _r  > 10:
                _repeating = 10
            else:
                _repeating = repeating - _r

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    temperature=temperature,  # default temperature is 0.7
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=_repeating,
                    do_sample=True
                )
            for i in range(0, len(generated_ids), _repeating):
                # Extract sequences corresponding to each input prompt
                chunk = generated_ids[i:i + _repeating]
                input_ids = model_inputs.input_ids[i // _repeating]
                for ids in chunk:
                    response = tokenizer.decode(ids[len(input_ids):], skip_special_tokens=True)
                    responses_list[i // _repeating].append(response)

        responses = responses_list[0]
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                temperature=temperature,  # default temperature is 0.7
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode all responses in the batch
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses = [response]

    # print('Log, responses:', responses)

    return responses


def custom_vision_generator(model, messages, image, max_tokens=1024, temperature=0.7, repeating=1, **kwargs):
    if 'gemma' in model:
        if messages[0]['role'] == 'system':
            messages = messages[1:]
    model, processor = get_model_by_name(model)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print('########## Log, ')
    # print(image)
    # print(text)
    model_inputs = processor(images=[image], text=[text], return_tensors="pt", padding=True, padding_side='left',
                             truncation=True).to(model.device)

    if repeating > 1:
        bs = 1
        responses_list = [[] for _ in range(bs)]  # no [[]]* bs, the inner [] is shared
        # 修改为每一个batch最大10，否则容易OOM
        for _r in range(0, repeating, 10):
            if repeating - _r  > 10:
                _repeating = 10
            else:
                _repeating = repeating - _r

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    temperature=temperature,  # default temperature is 0.7
                    max_new_tokens=max_tokens,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    num_return_sequences=_repeating,
                    do_sample=True
                )
            for i in range(0, len(generated_ids), _repeating):
                # Extract sequences corresponding to each input prompt
                chunk = generated_ids[i:i + _repeating]
                input_ids = model_inputs.input_ids[i // _repeating]
                for ids in chunk:
                    response = processor.decode(ids[len(input_ids):], skip_special_tokens=True)
                    responses_list[i // _repeating].append(response)

        responses = responses_list[0]
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                temperature=temperature,  # default temperature is 0.7
                max_new_tokens=max_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=True
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode all responses in the batch
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses = [response]

    # print('Log, responses:', responses)
    # 多个返回值，采样
    return responses


def compute_combination_by_budget(total_budget):
    # total_budget = 3
    retrieve_model_list = [72, 32, 7]
    generate_model_list = [70, 8, 3]
    t = {}
    for r in retrieve_model_list:
        for g in generate_model_list:
            unit1 = 4 * (72 // r) - 3
            unit2 = 4 * (70 // g) - 3
            top_times = total_budget * unit1
            for i in range(1, top_times + 1):
                remaining1 = (total_budget * unit1 - i) // unit1
                top_times2 = remaining1 * unit2
                if top_times2 > 0:
                    t[(r, i, g)] = top_times2
                    print(f'retrieve {r}B', i, f'times. gen model {g}B', top_times2, 'times.')
    print()
    for k, v in t.items():
        print(f'retrieve {k[0]}B', k[1], f'times. gen model {k[2]}B', v, 'times.')
