from camel.typing import ModelType


global_arg2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
                 'GPT_4': ModelType.GPT_4,
                #  'GPT_4_32K': ModelType.GPT_4_32k,
                 'GPT_4_TURBO': ModelType.GPT_4_TURBO,
                #  'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V
                'GPT_4O': ModelType.GPT_4O,
                'GPT_4O_MINI': ModelType.GPT_4O_MINI,  # is also a string
                 'llama1b': ModelType.LLAMA_1B,
                 'llama3b': ModelType.LLAMA_3B,
                 'llama8b': ModelType.LLAMA_8B,
                 'llama70b': ModelType.LLAMA_70B,
                 'llama405b': ModelType.LLAMA_405B,
                 'qwen3b': ModelType.QWEN_3B,
                 'qwen7b': ModelType.QWEN_7B,
                 'qwen14b': ModelType.QWEN_14B,
                 'qwen32b': ModelType.QWEN_32B,
                 'qwen72b': ModelType.QWEN_72B,
                 }


global_num_max_token_map = {
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


MODEL_TYPE = {
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
        ModelType.STUB,
        # None
    }



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
