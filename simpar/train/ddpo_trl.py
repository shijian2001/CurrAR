#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate.utils import set_seed
from transformers.pipelines import pipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline, ModelConfig, TrlParser

from simpar.grpo.configs import CurriculumConfig
from simpar.train.curriculum import Curriculum
from simpar.train.curriculum_dataloader import CurriculumDataLoader
from simpar.train.scorer import VQAScorer

logger = logging.getLogger(__name__)


class CurriculumManager:
    """课程学习管理器，用于动态调整数据难度"""

    def __init__(self, curriculum: Curriculum, data_loader: CurriculumDataLoader):
        self.curriculum = curriculum
        self.data_loader = data_loader
        self.last_difficulty = 0

    def update_difficulty(self, reward, current_step):
        """根据奖励和当前步骤更新数据难度"""
        self.last_difficulty = self.curriculum.infer_target_difficulty(
            {"current_step": current_step, "difficulty": self.last_difficulty, "reward": reward}
        )
        self.data_loader.set_difficulty(self.last_difficulty)
        return self.last_difficulty


def create_vqa_reward_function(vqa_pipeline, curriculum_manager, state):
    """创建VQA奖励函数"""

    scorer = VQAScorer()

    def reward_function(samples, prompts, metadata):
        """
        计算生成样本的奖励

        Args:
            samples: 生成的样本tokens
            prompts: 原始提示

        Returns:
            rewards: 奖励分数
        """

        # 使用VQA模型评估图像质量
        rewards, meta = scorer.calc_score(vqa_pipeline, samples, prompts, metadata=metadata)

        # 更新课程学习难度
        avg_reward = rewards.mean()
        current_step = state.global_step
        difficulty = curriculum_manager.update_difficulty(avg_reward, current_step)

        # 记录指标
        logger.info(f"Step {current_step}, Average Reward: {avg_reward:.4f}, Difficulty: {difficulty}")

        return rewards, meta

    return reward_function


@dataclass
class DDPOScriptArguments:
    """
    DDPO脚本参数
    """

    prompt_filename: str
    pretrained_model: str = field(default="runwayml/stable-diffusion-v1-5")
    vqa_model: str = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "模型检查点路径"})


# 创建trainer state对象来跟踪全局步骤
class TrainerState:
    def __init__(self):
        self.global_step = 0


class StatedTrainer(DDPOTrainer):
    def __init__(self, state: TrainerState, curriculum_manager: CurriculumManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.curriculum_manager = curriculum_manager
        curriculum_manager.data_loader.init(self.accelerator, kwargs["config"].train_batch_size)

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)
            self.state.global_step = global_step


def main(
    script_args: DDPOScriptArguments,
    training_args: DDPOConfig,
    model_args: ModelConfig,
    curriculum_args: CurriculumConfig,
):
    # 设置随机种子以确保可重复性
    set_seed(training_args.seed)

    ###############
    # 设置日志
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 使用CurriculumDataLoader加载数据集
    data_loader = CurriculumDataLoader(prompt_path=script_args.prompt_filename)

    # 设置模型参数
    logger.info("*** 初始化模型参数 ***")

    # 创建课程学习管理器
    curriculum = Curriculum(
        difficulty_range_getter=data_loader.get_difficulty_range,
        eta=curriculum_args.eta,
        beta=curriculum_args.c_beta,
        alpha=curriculum_args.alpha,
        strategy=curriculum_args.curriculum_strategy,
        sample_num_batches_per_epoch_getter=data_loader.get_sample_num_batches_per_epoch,
    )

    # TODO: use dataloader itself
    curriculum_manager = CurriculumManager(curriculum, data_loader)

    # 创建VQA管道
    vqa_pipeline = pipeline(
        "image-text-to-text",
        model=script_args.vqa_model,
        torch_dtype=torch.bfloat16,
        batch_size=training_args.sample_batch_size,
        device_map="auto",
    )
    vqa_pipeline.model.eval()

    state = TrainerState()

    # 创建奖励函数
    reward_function = create_vqa_reward_function(
        vqa_pipeline=vqa_pipeline,
        curriculum_manager=curriculum_manager,
        state=state,
    )
    sd_pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
    )

    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        # "total_limit": 5,
        "project_dir": "./save",
    }

    # 初始化DDPO训练器
    trainer = StatedTrainer(
        curriculum_manager=curriculum_manager,
        config=training_args,
        sd_pipeline=sd_pipeline,
        reward_function=reward_function,
        state=state,
        prompt_function=curriculum_manager.data_loader.prompt_loader.next,
    )

    # 初始化data_loader
    data_loader.init(trainer.accelerator, training_args.train_batch_size)

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DDPOScriptArguments, DDPOConfig, ModelConfig, CurriculumConfig))
    assert sys.argv[-1].endswith("yml") or sys.argv[-1].endswith("yaml")
    script_args, training_args, model_args, curriculum_args = parser.parse_yaml_file(sys.argv[-1])
    main(script_args, training_args, model_args, curriculum_args)
