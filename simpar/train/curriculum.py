import json
import random
from pathlib import Path
from typing import Any, Callable

import accelerate
import numpy as np
import tqdm
from loguru import logger


class Curriculum:
    def __init__(
        self,
        sample_num_batches_per_epoch_getter: Callable[[], int],
        difficulty_range_getter: Callable[[], tuple[int, int]],
        eta: float,
        alpha: float,
        beta: float,
        strategy: str = "random",
    ):
        self.strategy = strategy
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.strategy = strategy
        self.sample_num_batches_per_epoch_getter = sample_num_batches_per_epoch_getter
        self.difficulty_range_getter = difficulty_range_getter

    def infer_target_difficulty(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        if self.strategy == "random":
            # TODO: 需要传进来难度scale，或者我们后面定好
            return random.randint(min_difficulty, max_difficulty)
        elif self.strategy == "reward":
            return self._reward_based_infer(metadata)
        elif self.strategy == "timestep":
            return self._timestep_based_infer(metadata)
        else:
            raise NotImplementedError("Not implemented yet")

    def _reward_based_infer(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        result_difficulty = metadata["difficulty"] + self.eta * np.tanh(self.alpha * metadata["reward"] - self.beta)
        return min(max(0, result_difficulty) + min_difficulty, max_difficulty)

    def _timestep_based_infer(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        return min(
            max(
                0,
                metadata["current_step"]
                // (self.sample_num_batches_per_epoch_getter() / (max_difficulty - min_difficulty + 1)),
            )
            + min_difficulty,
            max_difficulty,
        )


class CurriculumPromptLoader:
    def __init__(self, prompt_path: str) -> None:
        self.accelerator: None | accelerate.Accelerator = None
        self.difficulty_to_prompts: dict[int, list[dict[str, Any]]] = {}
        self.difficulty_to_prompts_idx: dict[int, int] = {}
        self.prompt_path = Path(prompt_path)
        self.current_difficulty = 1
        self.sample_num_batches_per_epoch = 0
        self.t: tqdm.tqdm | None = None
        self.difficulty_range: tuple[int, int] | None = None
        self.data_num = 0

    def get_sample_num_batches_per_epoch(self) -> int:
        return self.sample_num_batches_per_epoch

    def get_difficulty_range(self) -> tuple[int, int]:
        assert self.difficulty_range, "need init"
        return self.difficulty_range

    def init(self, accelerator: accelerate.Accelerator, batch_size: int):
        self.accelerator = accelerator
        total = 0
        logger.info(f"initial index: {self.accelerator.process_index}, num process: {self.accelerator.num_processes}")
        for difficulty_str, prompts in json.loads(self.prompt_path.read_text()).items():
            total += len(prompts)
            self.difficulty_to_prompts[self._extract_difficulty(difficulty_str)] = prompts
            self.difficulty_to_prompts_idx[self._extract_difficulty(difficulty_str)] = self.accelerator.process_index
        self.t = tqdm.tqdm(total=total, desc="dataloader")
        self.data_num = total
        self.sample_num_batches_per_epoch = total // (self.accelerator.num_processes * batch_size)
        self.difficulty_range = (min(self.difficulty_to_prompts), max(self.difficulty_to_prompts))

    def _extract_difficulty(self, difficulty_str: str) -> int:
        return int(difficulty_str.split("_")[-1])

    def next(self) -> tuple[str, Any]:
        assert self.accelerator and self.t, "not initialize"
        self.t.update(self.accelerator.num_processes)
        if self.difficulty_to_prompts_idx[self.current_difficulty] >= len(
            self.difficulty_to_prompts[self.current_difficulty]
        ):
            logger.warning(f"difficulty {self.current_difficulty} has no more prompts, reset to 0")
            self.difficulty_to_prompts_idx[self.current_difficulty] = self.accelerator.process_index
        prompt = self.difficulty_to_prompts[self.current_difficulty][
            self.difficulty_to_prompts_idx[self.current_difficulty]
        ]
        self.difficulty_to_prompts_idx[self.current_difficulty] += self.accelerator.num_processes
        return prompt["prompt"], prompt

    def set_difficulty(self, difficulty: int) -> None:
        logger.info(f"set difficulty to {difficulty}")
        self.current_difficulty = difficulty
