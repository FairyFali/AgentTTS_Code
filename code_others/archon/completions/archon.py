from .components import (
    Generator,
    Ranker,
    Fuser,
    Critic,
    Verifier,
    Unit_Test_Generator,
    Unit_Test_Evaluator,
    Component,
)
from . import utils
from loguru import logger
import random
from collections import defaultdict
from typing import Any, Tuple

COMPONENT_TYPE_MAP = {
    "generator": Generator,
    "ranker": Ranker,
    "fuser": Fuser,
    "critic": Critic,
    "verifier": Verifier,
    "unit_test_generator": Unit_Test_Generator,
    "unit_test_evaluator": Unit_Test_Evaluator,
}


class Layer:
    def __init__(self, layer_config: list, custom_components: dict, custom_generators: dict):
        """ Initialize the archon layer

        Args:
            layer_config (list): A list of dicts, where each dict is a component of the layer
            custom_components (list): A list of custom components
            custom_generators (list): A list of custom generators 
        """

        self.config = layer_config
        self.components = []
        self.custom_components = custom_components
        self.custom_generators = custom_generators
        self.initialize_layer()

    def initialize_layer(self):
        """Initialize the layer and its components"""

        component_list = self.config

        # deprecated compatibility with old configs
        if isinstance(self.config, dict):
            component_list = self.config["models"]

        # initialize each component in the layer
        for component_config in component_list:
            component_type = component_config["type"]  # an important field

            # try fpr supported
            # key part
            #
            #
            if component_type in COMPONENT_TYPE_MAP:
                if component_type == "generator":
                    # If the type is generator
                    # Could add own generators
                    self.components.append(
                        COMPONENT_TYPE_MAP[component_type](
                            component_config, self.custom_generators
                        )
                    )
                else:
                    # for other components
                    # I add the custom generators
                    self.components.append(
                        COMPONENT_TYPE_MAP[component_type](component_config, self.custom_generators)
                    )
            else:
                try:
                    print('log,', self.custom_components)
                    # try for custom
                    component = self.custom_components[component_config["type"]]
                    self.components.append(component(component_config, self.custom_generators))
                except Exception as e:
                    logger.error(e)
                    raise ValueError(
                        f"Unsupported object type: {component_config['type']}. Check config (set Custom to true), add custom component before initiliaziation, and make sure custom component has been correctly made"
                    )
        logger.info(f"Initialized layer with {len(self.components)} components")

    def process(self, conv: list, prev_state: dict):
        """ Have the layer process the conversation

        Args:
            conv (list): A list of the conversation so far
            prev_state (dict): The state of the previous layer

        Returns:
            dict: returns a new state to send to the next layer. 
                All unchanged values from the previous layer is passed to this layer
        """

        # default to list, although you can also store variables
        current_state = defaultdict(list)

        prev_candidates_len = len(prev_state["candidates"])  # number of previous candidates
        if prev_candidates_len > 32:
            logger.info(
                f"WARNING: Previous inputs of length ({prev_candidates_len}) are too long! Will likely exceed context window of generator LMs"
            )

        # could parallelize this
        for model in self.components:
            # print('### #')
            if isinstance(model, Component):  # Run Component
                # after running, the current_state will automatically be modified
                # the update to current_state is only for
                model.run(conv, prev_state, current_state)
            else:
                raise ValueError(f"Unsupported object type: {type(model).__name__}")

        for key in prev_state:
            if key not in current_state:  # if value has not been updated
                current_state[key] = prev_state[key]  # populate and pass to next layer

        return current_state


class Archon:
    """
    Archon class to generate responses given an input by applying multiple layers
    of inference times techniques sequentially.
    """

    def __init__(
        self, config: dict, api_key_data: Any=None, query_saves: bool=False, mock_api_calls: bool=False
    ):
        """ Initialize the Archon with configuration settings.

        Args:
            config (_type_): Configuration dictionary containing layers and other settings.
            api_key_data (Any, optional): api_key data to use on generation. Defaults to None.
            query_saves (bool, optional): save the queries generated by each layer for analysis. Defaults to False.
            mock_api_calls (bool, optional): generate mock responses instead of calling the model provider. Defaults to False.
        """

        self.config = config
        self.initialized = False
        self.mock_api_calls = mock_api_calls
        self.query_saves = query_saves

        # attributes for custom
        self.custom = config.get("custom", False)
        self.custom_components = {}
        self.custom_generators = {}

        # Attempts load from api_keys.json or os.environ
        utils.KEYS = utils.keyHandler(api_key_data)

        # if custom, user has to manually initialize
        if not self.custom:
            self.initialize()
        else:
            logger.warning(
                "Custom model, make sure to add custom components before initializing."
            )

    def add_component(self, name: str, component: Component):
        """add a custom component for use in archon configuration

        Args:
            name (str): Name of component, must match name in archon config
            component (Component): Component to be called during inference time
        """
        self.custom_components[name] = component

    def add_generator(self, name: str, generator):
        """add a custom generator for use in archon configuration

        Args:
            name (str): Name of generator, must match name in archon config
            generator (): generator function to be called from a generator
        """
        self.custom_generators[name] = generator

    def initialize(self):
        """
        Initialize the archon model, layer by layer.
        """

        self.layers = []
        for layer_config in self.config["layers"]:
            # each layer_config is a list with same type
            # before initialize, if we would like to add own component or generator, we need run add_component/add_generator first.
            layer = Layer(layer_config, self.custom_components, self.custom_generators)
            self.layers.append(layer)

        print(f"Archon initialized with {len(self.layers)} layers.")
        self.initialized = True

    def generate(self, conv: list) -> str:
        """generate a single output to the latest query in the conversation using your Archon config.

        Args:
            conv (list): A list of the conversation so far

        Returns:
            str: generated answer to given conversation
        """
        if self.mock_api_calls:
            return "Mock Inference API Response"

        if not self.initialized:
            raise Exception(
                f"Initialize your archon model before generating. This most likely happens because you have a custom component"
            )

        # if only query was given
        if isinstance(conv, str):
            conv = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": conv},
            ]
        elif conv[0]["role"] != "system":  # add a system message if missing
            conv = [{"role": "system", "content": "You are a helpful assistant"}] + [
                message for message in conv
            ]

        responses, output_storage = self._generate(conv)

        if utils.DEBUG_ARCHON:
            if not len(responses) > 0:
                logger.error(f"responses is empty: {responses}")
            if not isinstance(responses, list):
                logger.error(f"responses is not a list: {responses}")
            if not isinstance(responses[0], str):
                logger.error(
                    f"First element of responses is not a string: {responses[0]}"
                )

        if self.query_saves:
            import os
            import json
            from datetime import datetime

            # Create the directory if it doesn't exist
            save_dir = os.path.join("outputs", "query_saves", self.config["name"])
            os.makedirs(save_dir, exist_ok=True)

            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['name']}_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            # Save output_storage to the file
            with open(filepath, "w") as f:
                json.dump(output_storage, f, indent=2)

            print(f"Output saved to: {filepath}")

        assert (
            len(responses) > 0
            and isinstance(responses[0], str)
            and isinstance(responses, list)
        ), f"response not valid: {responses}"

        # random if multiple outputs
        if len(responses) > 1:
            responses = [random.choice(responses)]

        return responses[0]

    def _generate(self, conv: list) -> Tuple[list, list]:
        """ Generate responses by applying each layer sequentially to the inputs.
        
        Args:
            conv (list): List of the conversation so far

        Returns:
            Tuple[list, list]: a tuple of final_outputs and output_storage. 
                A list containing the final response and a list of data to store
        """        

        # messages should just be a single list of optional system and user messages
        # query = conv[-1]["content"]

        prev_state = defaultdict(list)  # previous candidates
        output_storage = []

        for i in range(len(self.layers)):

            layer = self.layers[i]

            if utils.DEBUG:
                prev_candidates = prev_state["candidates"]
                prev_critiques_len = len(prev_state["critiques"])
                logger.debug(
                    f"Running layer {i}, with {len(prev_candidates)} previous candidates and {prev_critiques_len} previous critiques"
                )

            # key part
            new_state = layer.process(
                conv,
                prev_state,
            )

            # This is prob how I would want to do query saves later.
            # Save the states, and prob have a verbose for each one
            # print(f"-----PREV_{i}-----------")
            # for key in prev_state.keys():
            #     print(f"{key}:{len(prev_state[key])}")
            # print(f"-----NEW_{i}-------------")
            # for key in new_state.keys():
            #     print(f"{key}:{len(new_state[key])}")
            # print("-----------------")

            prev_state = new_state

            if self.query_saves:
                current_output = []
                for i, component_config in enumerate(layer.config):
                    component_config_with_output = component_config.copy()
                    component_config_with_output["output"] = prev_state["candidates"][i]
                    component_config_with_output["critique"] = prev_state["critiques"]
                    current_output.append(component_config_with_output)
                output_storage.append(current_output)

        final_outputs = prev_state["candidates"]

        # print('Log final outputs:', final_outputs)

        if len(prev_state["candidates"]) == 0:
            logger.warning("No output generated by Archon!")
        elif len(prev_state["candidates"]) > 1:
            # if the output candiates is more than one, return the random one.
            prev_candidates = prev_state["candidates"]
            logger.warning(
                f"Multiple outputs generated by Archon! Returning a random candidate from the set of {prev_candidates} choices."
            )
            final_outputs = [random.choice(final_outputs)]

        return final_outputs, output_storage
