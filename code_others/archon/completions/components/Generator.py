from ..utils import (
    generate_together,
    generate_openai,
    generate_anthropic,
    generate_groq,
    generate_google,
    generate_tgi,
    generate_bedrock,
    vllmWrapper,
    clean_messages,
)
import loguru as logger

from .Component import Component

GENERATE_MAP = {
    "Together_API": generate_together,
    "OpenAI_API": generate_openai,
    "Anthropic_API": generate_anthropic,
    "Groq_API": generate_groq,
    "Google_API": generate_google,
    "tgi": generate_tgi,
    "Bedrock_API": generate_bedrock,
}


class Generator(Component):
    def __init__(self, config, custom_generators=None):
        """
        Initialize the Model with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.custom_generators = custom_generators
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the model and tokenizer with the specified settings.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.samples = self.config.get("samples", 1)  # config could add "samples", 1 by default
        self.no_system = self.config.get("no_system", False)

        if self.model_type in GENERATE_MAP:
            self.generator = GENERATE_MAP[self.model_type]
        elif self.model_type == "vLLM":
            self.generator = vllmWrapper(self.model_name)
        else:
            # print('Log,', self.custom_generators)
            try:
                # trying for custom
                self.generator = self.custom_generators[self.model_type]
            except Exception as e:
                print(e)
                raise ValueError(
                    f"Invalid model type: {self.model_type}. Check config (set Custom to true), \
                        add custom generator before initiliaziation, and make sure custom generator has been correctly made"
                )

        print(f"Model initialized: {self.model_name}")

    def generate_from_messages(self, messages, image=None, temperature=None):
        """
        no support batch
        Generate a response based on the input text.

        Parameters:
        messages to get a response with

        Returns:
        list of str: The generated responses.
        """
        if temperature is None:
            temperature = self.temperature

        if self.no_system:  # remove system message
            messages = messages[1:]

        # Cap output to model max context length - input context length
        # parsed_content = ' '.join([msg['content'] for msg in clean_messages(messages)])
        # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # estimate since not all models have public tokenizers
        # input_token_len = len(encoding.encode(parsed_content, disallowed_special=())) - 5 # buffer for model variance
        # max_tokens = self.max_context_length - input_token_len

        max_tokens = self.max_tokens

        if self.model_type in GENERATE_MAP:
            outputs = []
            for _ in range(self.samples):
                # print('### 1')
                output = self.generator(
                    model=self.model_name,
                    messages=clean_messages(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if output is not None:
                    outputs.append(output)
        else:
            # for customized generator
            # for model_type in GENERATE_MAP, it generate one in one time.
            # for ours, we use the repeating so generate multiple in one time.
            # print('Log:', messages)
            # print('generator', self.generator)
            # print('### 2')
            outputs = self.generator(
                model=self.model_name,
                messages=clean_messages(messages),
                image = image,
                max_tokens=max_tokens,
                temperature=temperature,
                repeating=self.samples
            )

        return outputs

    def run(self, conversation, prev_state, state):
        """
        Run a component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon. 
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """
        # print("### 3")
        outputs = self.generate_from_messages(conversation)
        
        state["candidates"].extend(outputs)

        return

    def generate_from_query(self, query: str = None, temperature=None):
        assert isinstance(query, str)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        return self.generate_from_messages(messages, temperature)
