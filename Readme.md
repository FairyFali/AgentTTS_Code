### Anonymous Code and Data for AgentTTS

This repository contains the anonymous code and data used in AgentTTS experiments.

#### `data/`

This folder contains datasets from four types of tasks used for evaluation.

#### `code_chatdev/`

This folder includes the code adapted from the paper *Communicative Agents for Software Development* (ChatDev).
We gratefully acknowledge their open-source contribution.

* The main entry point is `run_repeating.py`, which runs AgentTTS on ChatDev-style tasks.

#### `code_others/`

This folder contains code for running AgentTTS on other benchmark datasets.
It is adapted from the open-source implementation of the paper *Archon: An Architecture Search Framework for Inference-Time Techniques*.

* Each dataset has a dedicated entry script in the format `archon_[dataset_name].py`, which runs the main TTS procedure.
* `AgentTTS.py` is the central entry point for invoking AgentTTS logic across these datasets.
