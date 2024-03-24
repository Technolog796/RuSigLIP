import random
from typing import Iterator

import torch
from torch import Tensor

from .wiki_dataset import RuSigLIPDataset


class SigLIPDataLoader:
    def __init__(self, dataset: RuSigLIPDataset, batch_size: int, rank: int, world_size: int, seed: int = 42):
        self.dataset = dataset

        self.batch_size = batch_size
        self.chunk_size = batch_size // world_size

        self.rank = rank
        self.world_size = world_size

        self.epoch = 0
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, list[dict[str, Tensor]]]]:
        indices = self._get_indices()
        languages = self._get_languages()

        for batch_idx in range(len(self)):
            start = batch_idx * self.batch_size
            split = start + self.chunk_size * self.rank
            end = start + self.batch_size

            batch_indices = torch.concat([indices[split:end], indices[start:split]])
            batch_indices = batch_indices.split(self.chunk_size)
            batch_language = languages[batch_idx]

            yield self._get_batch(batch_indices, batch_language)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_batch(self, indices: Tensor, language: str) -> tuple[Tensor, list[dict[str, Tensor]]]:
        images = torch.stack([self.dataset.get_image(idx) for idx in indices[0]])
        texts = [self.dataset.get_texts(indices[chunk_idx])[language] for chunk_idx in range(self.world_size)]
        return images, texts

    def _get_indices(self) -> Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g)
        return indices

    def _get_languages(self) -> list[str]:
        random.seed(self.seed + self.epoch)
        languages = [random.choice(["en", "ru"]) for _ in range(len(self))]
        return languages