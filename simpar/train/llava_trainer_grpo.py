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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Union

import datasets
import torch
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from transformers import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad

from simpar.grpo.configs import CurriculumConfig, GRPOConfig
from simpar.grpo.utils.callbacks import get_callbacks
from simpar.grpo.utils.wandb_logging import init_wandb_training
from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.train.curriculum import Curriculum
from simpar.train.curriculum_dataloader import CurriculumDataLoader
from simpar.train.scorer import OpenAIVQAScorer, VQAScorer

logger = logging.getLogger(__name__)


class LLaVAGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        curriculum: Curriculum,
        vqa_model_name: str,
        data_loader: CurriculumDataLoader,
        scorer_type: str = "local",
        *args,
        **kwargs,
    ):
        # 删除kwargs中的train_dataset参数，传入get_dataloader获取的dataloader
        if "train_dataset" in kwargs:
            kwargs["train_dataset"] = data_loader.get_dataset()

        # 初始化父类
        super().__init__(*args, **kwargs)

        self.train_dataset = data_loader.get_dataset()
        self.prompt_dataloader = data_loader
        # 初始化data_loader
        data_loader.init(self.accelerator, self.args.per_device_train_batch_size)
        # 覆盖train_dataset为data_loader获取的dataloader

        # 初始化其他参数
        self.curriculum = curriculum
        self.last_difficulty = 0
        self.scorer_type = scorer_type

        # 只有在使用本地模型时才初始化vqa_pipeline
        if scorer_type == "local":
            self.vqa_pipeline = pipeline(
                "image-text-to-text",
                model=vqa_model_name,
                torch_dtype=torch.bfloat16,
                batch_size=self.args.per_device_train_batch_size,
            )
            self.vqa_pipeline.model.eval()
        else:
            # 对于API模式，不需要本地pipeline
            self.vqa_pipeline = None

    def _decode_images(self, completion_ids):
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = torch.stack(completion_ids, dim=0)

        codebook_size = 64000
        latent_size = 1024 // 16
        index_samples = completion_ids - len(self.processing_class)
        index_samples = torch.clamp(index_samples, min=0, max=codebook_size - 1)
        index_samples = index_samples.reshape(-1, latent_size, latent_size).unsqueeze(1)

        with torch.inference_mode():
            generated_images = self.vq_model.decode(index_samples).squeeze(2)

        # resize to 224 to save memory
        generated_images = torch.nn.functional.interpolate(
            generated_images, size=(224, 224), mode="bilinear", align_corners=False
        )
        generated_images = (255 * (generated_images * 0.5 + 0.5)).clamp(0, 255)

        mean = torch.tensor(OPENAI_DATASET_MEAN, device=device)
        std = torch.tensor(OPENAI_DATASET_STD, device=device)

        transformed_images = generated_images / 255.0  # B, 3, 224, 224
        transformed_images = (transformed_images - mean[None, :, None, None]) / std[None, :, None, None]

        # convert to list for broadcast
        transformed_images = [img.cpu() for img in transformed_images]

        return transformed_images

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        metadata = [x["metadata"] for x in inputs]
        prompts_text = [p for p in prompts]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                if len(ordered_set_of_prompts) < 7:
                    ordered_set_of_prompts = (
                        ordered_set_of_prompts + ordered_set_of_prompts[: 7 - len(ordered_set_of_prompts)]
                    )

                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)

                decoded_images = self._decode_images(completion_ids)  # List of images [C, H, W]

            else:
                completion_ids = [None] * len(all_prompts_text)
                decoded_images = [None] * len(all_prompts_text)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            decoded_images = broadcast_object_list(decoded_images, from_process=0)
            decoded_images = decoded_images[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            raise NotImplemented("not supported yet")
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            output_reward_func = reward_func(
                vqa_pipeline=self.vqa_pipeline, images=decoded_images, prompts=prompts, metadata=metadata
            )
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        self.last_difficulty = self.curriculum.infer_target_difficulty(
            {
                "current_step": self.state.global_step,
                "difficulty": self.last_difficulty,
                "reward": rewards.mean().cpu().numpy(),
            }
        )
        self.prompt_dataloader.set_difficulty(self.last_difficulty)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

    vq_model_ckpt: str = field(default="/path_to_tokenizer/Cosmos-1.0-Tokenizer-DV8x16x16")


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
    curriculum_args: CurriculumConfig,
):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)

    # Load VQ model
    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    vq_model = CosmosTokenizer(
        checkpoint_enc=f"{script_args.vq_model_ckpt}/encoder.jit",
        checkpoint_dec=f"{script_args.vq_model_ckpt}/decoder.jit",
        tokenizer_config=tokenizer_config,
    )
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Load the dataset using CurriculumDataLoader
    data_loader = CurriculumDataLoader(prompt_path=training_args.prompt_path)

    # Get reward functions - 根据配置选择不同的scorer类型
    if training_args.vqa_scorer_type == "openai":
        logger.info(f"使用OpenAI兼容API scorer: {training_args.vqa_api_base_url}")
        scorer_ = OpenAIVQAScorer(
            api_base_url=training_args.vqa_api_base_url,
            api_key=training_args.vqa_api_key,
            model_name=training_args.vqa_api_model_name,
            timeout=training_args.vqa_api_timeout,
        )
    else:
        logger.info(f"使用本地VQA模型: {training_args.vqa_model_name}")
        scorer_ = VQAScorer()

    REWARD_FUNCS_REGISTRY = {
        "vqa": scorer_.calc_score,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    curriculum = Curriculum(
        difficulty_range_getter=data_loader.get_difficulty_range,
        eta=curriculum_args.eta,
        beta=curriculum_args.c_beta,
        alpha=curriculum_args.alpha,
        strategy=curriculum_args.strategy,
        sample_num_batches_per_epoch_getter=data_loader.get_sample_num_batches_per_epoch,
    )

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = LLaVAGRPOTrainer(
        vqa_model_name=training_args.vqa_model_name,
        curriculum=curriculum,
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        data_loader=data_loader,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
        scorer_type=training_args.vqa_scorer_type,
    )
    trainer.vq_model = vq_model

    # trainer.aesthetic_model = aest_model

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # 注意这里不再需要dataset[script_args.dataset_train_split]
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, CurriculumConfig))
    script_args, training_args, model_args, curriculum_args = parser.parse_args_and_config()
    assert training_args.prompt_path, "require prompt_path"
    main(script_args, training_args, model_args, curriculum_args)
