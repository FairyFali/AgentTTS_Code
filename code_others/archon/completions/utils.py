import copy
import requests
import time
import os
from loguru import logger
import openai
import anthropic
from groq import Groq
import google.generativeai as google_genai
import json
import random
from litellm import completion
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, BitsAndBytesConfig
from transformers import MllamaForConditionalGeneration
import torch.nn.functional as F
import torch

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))
DEBUG_VERIFIER = int(os.environ.get("DEBUG_VERIFIER", 0))
DEBUG_ARCHON = int(os.environ.get("DEBUG_VERIFIER", 0))
DEBUG_UNIT_TEST_GENERATOR = int(os.environ.get("DEBUG_UNIT_TEST_GENERATOR", 0))
DEFAULT_CONFIG = "configs/archon-1-110bFuser-2-110bM.json"

KEYS = None  # Initialized in Archon

KEY_NAMES = (
    "OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION_NAME",
    "GOOGLE_API_KEY",
    "EXAMPLE_API_KEY",
)


class AllKeysUsedError(Exception):
    """Raised when all available API keys for a specific type have been used."""

    pass


class keyHandler:
    def __init__(self, api_key_data: None):
        self.api_key_data = api_key_data

        if api_key_data:
            self.all_api_keys = self._load_api_keys()
        else:
            self.all_api_keys = self._load_env_keys()

        self.key_indices = {key: 0 for key in self.all_api_keys}

    def _load_api_keys(self):
        if isinstance(self.api_key_data, dict):
            return self.api_key_data
        else:
            try:
                with open(self.api_key_data, "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"API key file '{self.api_key_data}' not found."
                )

    def _load_env_keys(self) -> Dict[str, Any]:
        # Dictionary to store the processed environment variables
        processed_env: Dict[str, List[str]] = {}

        # Regular expression to match the variable names
        pattern = re.compile(r"^(.+?)(?:_(\d+))?$")

        # Iterate through all environment variables
        for key, value in os.environ.items():
            # Check if the key starts with the prefixes we're interested in
            if key.startswith(KEY_NAMES):
                match = pattern.match(key)
                if match:
                    base_key, _ = match.groups()

                    if base_key in processed_env:
                        processed_env[base_key].append(value)
                    else:
                        processed_env[base_key] = [value]

        return processed_env

    def get_current_key(self, api_key_type):
        if api_key_type not in self.all_api_keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Make sure to add {api_key_type} to your .env file"
            )

        keys = self.all_api_keys[api_key_type]
        if not keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Make sure to add {api_key_type} to your .env file"
            )

        return keys[self.key_indices[api_key_type]]

    def switch_api_keys(self, api_key_type, api_key):
        print("switching key")
        if api_key_type not in self.all_api_keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Make sure to add {api_key_type} to your .env file"
            )

        keys = self.all_api_keys[api_key_type]
        if not keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Make sure to add {api_key_type} to your .env file"
            )

        current_index = keys.index(api_key)
        print(current_index)
        # used exhausted key, most likely a behind worker
        if current_index < self.key_indices[api_key_type]:
            print("used exhausted key, most likely a behind worker")
            return keys[self.key_indices[api_key_type]]

        new_index = current_index + 1

        if new_index == len(keys):
            print(f"No more keys to switch to")
            raise AllKeysUsedError(f"used all keys")

        self.key_indices[api_key_type] = new_index

        new_key = keys[new_index]
        print(
            f"Switched key for {api_key_type} from {keys[current_index]} to {new_key}"
        )
        return new_key


def clean_messages(messages):
    messages_alt = messages.copy()
    for msg in messages_alt:
        if isinstance(msg["content"], dict) and "content" in msg["content"]:
            msg["content"] = msg["content"]["content"]
    return messages_alt


def load_config(config_path):
    """
    Load the configuration from a given file path.
    If no path is provided or the file doesn't exist, use the default configuration.
    """
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            config_file = json.load(file)
            return config_file
    else:
        raise ValueError(
            f"config_path points to missing file. Reimport {config_path} to config directory"
        )


def format_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role.capitalize()}: {content}\n\n"
    prompt += "Assistant: "
    return prompt


class vllmWrapper:
    def __init__(self, model_name):
        from vllm import LLM
        from transformers import AutoTokenizer

        if DEBUG:
            logger.debug("Initializing vLLM model")
        self.model = LLM(model=model_name)

        if DEBUG:
            logger.debug("Initializing vLLM tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, model_name, messages, max_tokens, temperature, **kwargs):
        from vllm import SamplingParams

        if DEBUG:
            logger.debug(
                f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model_name}` with temperature {temperature}."
            )

        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        else:
            logger.info("No chat template, formatting as seen in util")
            prompt = format_prompt(messages)

        if DEBUG:
            logger.debug(f"Full prompt being sent: {prompt}")

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        if DEBUG:
            logger.debug(f"Output: `{response[:50]}...`.")

        return response


def generate_together(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    output = None

    key = (
        KEYS.get_current_key("TOGETHER_API_KEY")
        if KEYS
        else os.environ.get("TOGETHER_API_KEY")
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        res = None

        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}` with temperature {temperature}."
                )
                logger.debug(f"Full message being sent: {messages}")

            endpoint = "https://api.together.xyz/v1/chat/completions"

            time.sleep(2)

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {key}",
                },
            )
            if "error" in res.json():

                print("------------------------------------------")
                print(f"Model with Error: {model}")
                print(res.json())
                print("------------------------------------------")

                if res.json()["error"]["type"] == "invalid_request_error":
                    return None
                if res.json()["error"]["type"] == "credit_limit":
                    try:
                        key = KEYS.switch_api_keys("TOGETHER_API_KEY", key)
                        print(f"Retry in {sleep_time}s..")
                        time.sleep(sleep_time)
                        continue
                    except AllKeysUsedError as e:
                        logger.error(f"Exhausted all keys for Together")
                        break

            output = res.json()["choices"][0]["message"]["content"]

            break
        except AllKeysUsedError as e:
            logger.error(f"Exhausted all keys for Together")
            break
        except Exception as e:
            response = "failed before response" if res is None else res
            logger.error(f"{e} on response: {response}")
            print(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    if DEBUG:
        logger.debug(f"Output: `{output[:50]}...`.")

    return output.strip()


def generate_openai(model, messages, max_tokens=2048, temperature=0.7, **kwargs):

    key = (
        KEYS.get_current_key("OPENAI_API_KEY")
        if KEYS
        else os.environ.get("OPENAI_API_KEY")
    )

    client = openai.OpenAI(api_key=key)

    for sleep_time in [1, 2, 4, 8, 16, 32, 64]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

                logger.debug(f"Full message being sent: {messages}")
            if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_anthropic(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    key = (
        KEYS.get_current_key("ANTHROPIC_API_KEY")
        if KEYS
        else os.environ.get("ANTHROPIC_API_KEY")
    )
    client = anthropic.Client(api_key=key)

    time.sleep(2)
    output = None
    max_tokens = 4096
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            system = ""
            for message in messages:
                if message["role"] == "system":
                    system = message["content"]
                    break

            if system == "":
                logger.warning("No system message")

            messages_alt = [msg for msg in messages if msg["role"] != "system"]
            completion = client.messages.create(
                model=model,
                system=system,
                messages=messages_alt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.content[0].text
        except anthropic.RateLimitError as e:
            logger.error(e)
            if KEYS:
                try:
                    key = KEYS.switch_api_keys("ANTHROPIC_API_KEY", key)
                except AllKeysUsedError as e:
                    logger.error(f"Exhausted all keys for Anthropic")
                    break

                client = anthropic.Client(
                    api_key=key,
                )
            else:
                logger.info(f"Retry in {sleep_time}s..")
                time.sleep(sleep_time)
        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return None

    output = output.strip()

    return output


def generate_groq(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = Groq(
        api_key=KEYS.get_current_key("GROQ_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return output.strip()


def generate_tgi(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    client = openai.OpenAI(base_url=model, api_key="-")  # TGI endpoint

    output = client.chat.completions.create(
        model="tgi",
        messages=clean_messages(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    return output.strip()


def convert_gemini_format(messages):
    gemini_messages = []
    system = ""

    for msg in messages:
        if system == "" and msg["role"] == "system":
            system = msg["content"]
            continue

        message_type = "user"
        if msg["role"] == "assistant":
            message_type = "model"

        current_message = {"role": message_type, "parts": [{"text": msg["content"]}]}
        gemini_messages.append(current_message)

    if system == "":
        logger.warning("No system message")

    return system, gemini_messages


def generate_google(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    google_genai.configure(api_key=KEYS.get_current_key("GOOGLE_API_KEY"))

    system, messages_gemini = convert_gemini_format(messages)

    generation_config = google_genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        candidate_count=1,
    )

    if system != "":
        client = google_genai.GenerativeModel(model, system_instruction=system)
    else:
        client = google_genai.GenerativeModel(model)

    time.sleep(5)
    output = None

    for sleep_time in [4, 8, 16, 32, 64, 128, 256, 512]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            response = client.generate_content(
                messages_gemini, generation_config=generation_config
            )
            output = response.text
            break

        except Exception as e:
            logger.error(e)

            if sleep_time >= 32:
                sleep_time = sleep_time + random.randint(0, 16)

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if isinstance(output, str):
        output = output.strip()

    return output


def generate_bedrock(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            # Ensure that AWS credentials and region are set via environment variables
            os.environ["AWS_ACCESS_KEY_ID"] = KEYS.get_current_key("AWS_ACCESS_KEY_ID")
            os.environ["AWS_SECRET_ACCESS_KEY"] = KEYS.get_current_key(
                "AWS_SECRET_ACCESS_KEY"
            )
            os.environ["AWS_REGION_NAME"] = KEYS.get_current_key("AWS_REGION_NAME")

            # Call Bedrock model via litellm completion function
            response = completion(
                model=model,
                messages=clean_messages(messages),
            )

            output = response["choices"][0]["message"]["content"]
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return None

    return output.strip()


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


def load_pythia_model(model_name, device_map="auto"):
    '''
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
    #
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map=device_map, quantization_config=bnb_config, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #
    # return model, tokenizer
    pass # 因为tokenizer没有apply_chat_template方法


def load_mamba_model(model_name, device_map="auto"):
    '''
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



def load_starcoder_model(model_name, device_map="auto"):
    '''
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


def load_opencoder_model(model_name, device_map="auto"):
    '''
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
    # print("###Log, name", name, model_dict)
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
            '''
            3B 9T
            8B 15T
            70B 15T
            '''
            model, tokenizer = load_llama_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'gemma' in name:
            '''
            google/gemma-2-2b-it, 2T corpus
            google/gemma-2-9b-it, 8T corpus
            google/gemma-2-27b-it, 13T corpus
            '''
            if "gemma-3" in name:
                model, tokenizer = load_gemma3_model(name)
            else:
                model, tokenizer = load_gemma_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'Phi' in name:
            '''
            microsoft/Phi-3-mini-4k-instruct 3.8B, 3.3T
            microsoft/Phi-3-small-8k-instruct 7B, 4.8T 
            microsoft/Phi-3-medium-4k-instruct 14B, 4.8T
            '''
            model, tokenizer = load_phi_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'pythia' in name:
            '''
            EleutherAI/pythia-14m, 0.2T
            EleutherAI/pythia-70m
            EleutherAI/pythia-160m
            EleutherAI/pythia-410m
            EleutherAI/pythia-1b
            EleutherAI/pythia-1.4b
            EleutherAI/pythia-2.8b
            EleutherAI/pythia-6.9b
            EleutherAI/pythia-12b
            不是instruct
            '''
            model, tokenizer = load_pythia_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'mamba' in name:
            '''
            也没有apply template方法，舍弃
            "state-spaces/mamba-2.8b-hf"
            '''
            model, tokenizer = load_mamba_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'starcoder' in name:
            '''
            bigcode/starcoder2-3b
            bigcode/starcoder2-7b
            bigcode/starcoder2-15b
            不是instruct，没有apply template,舍弃
            '''
            model, tokenizer = load_starcoder_model(name)
            model_dict[name] = {}
            model_dict[name]['model'] = model
            model_dict[name]['tokenizer'] = tokenizer
            return model, tokenizer
        elif 'OpenCoder' in name:
            '''
            infly/OpenCoder-1.5B-Instruct
            infly/OpenCoder-8B-Instruct
            '''
            model, tokenizer = load_opencoder_model(name)
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
