import logging
from typing import Any, Dict, Iterator

from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset

from simpar.train.curriculum import CurriculumPromptLoader

logger = logging.getLogger(__name__)


class CurriculumIterableDataset(IterableDataset):
    """
    一个可迭代的数据集，包装了CurriculumPromptLoader
    """

    def __init__(self, prompt_loader: CurriculumPromptLoader):
        self.prompt_loader = prompt_loader
        self.accelerator = None
        self.batch_size = None

    def init(self, accelerator: Accelerator, batch_size: int):
        """初始化数据集，传递accelerator和batch_size"""
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.prompt_loader.init(accelerator, batch_size)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """返回一个迭代器，每次迭代返回一个样本"""
        assert self.accelerator is not None, "Dataset not initialized. Call init() first."
        while True:
            prompt, data = self.prompt_loader.next()
            yield {"prompt": prompt, "metadata": data}


class CurriculumDataLoader:
    """
    包装CurriculumPromptLoader的数据加载器，兼容Accelerate
    """

    def __init__(self, prompt_path: str):
        self.prompt_loader = CurriculumPromptLoader(prompt_path)
        self.dataset = CurriculumIterableDataset(self.prompt_loader)
        self.dataloader = None
        self.accelerator = None
        self.batch_size = None

    def init(self, accelerator: Accelerator, batch_size: int):
        """初始化数据加载器，传递accelerator和batch_size"""
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.dataset.init(accelerator, batch_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, examples):
        """将多个样本组合成一个批次"""
        batch = {}
        batch["prompt"] = [example["prompt"] for example in examples]
        batch["metadata"] = [example["metadata"] for example in examples]
        return batch

    def get_dataloader(self) -> DataLoader:
        """返回初始化后的数据加载器"""
        assert self.dataloader is not None, "DataLoader not initialized. Call init() first."
        return self.dataloader

    def set_difficulty(self, difficulty: int) -> None:
        """设置当前难度"""
        self.prompt_loader.set_difficulty(difficulty)

    def get_difficulty_range(self) -> tuple[int, int]:
        """获取难度范围"""
        return self.prompt_loader.get_difficulty_range()

    def get_sample_num_batches_per_epoch(self) -> int:
        """获取每个epoch的批次数"""
        return self.prompt_loader.get_sample_num_batches_per_epoch()

