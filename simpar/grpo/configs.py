# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(default_factory=lambda: [], metadata={"help": "The callbacks to run during training."})
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    prompt_path: str = field(default="", metadata={"help": "The path to the prompt."})
    vqa_model_name: str = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the pretrained model to use"})

    # VQA Scorer配置
    vqa_scorer_type: str = field(
        default="local",
        metadata={
            "help": "VQA scorer type: 'local' for local model inference, 'openai' for OpenAI-compatible API",
            "choices": ["local", "openai"],
        },
    )
    vqa_api_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL for OpenAI-compatible API when using 'openai' scorer type"},
    )
    vqa_api_key: str = field(
        default="dummy-key", metadata={"help": "API key for OpenAI-compatible API when using 'openai' scorer type"}
    )
    vqa_api_model_name: str = field(
        default="llava-v1.5-7b-hf", metadata={"help": "Model name to use with OpenAI-compatible API"}
    )
    vqa_api_timeout: int = field(
        default=30, metadata={"help": "Timeout in seconds for API requests when using 'openai' scorer type"}
    )


@dataclass
class CurriculumConfig:
    eta: float = field(default=50, metadata={"help": "The eta for the curriculum."})
    c_beta: float = field(default=0.5, metadata={"help": "The beta for the curriculum."})
    alpha: float = field(default=2, metadata={"help": "The alpha for the curriculum."})
    curriculum_strategy: str = field(default="reward", metadata={"help": "The strategy for the curriculum."})


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(default_factory=lambda: [], metadata={"help": "The callbacks to run during training."})
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
