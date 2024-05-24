import random
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor

from .dataset import RuSigLIPDataset


class SigLIPDataLoader:
    def __init__(self, dataset: RuSigLIPDataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.language = None

        self.size = len(dataset) // batch_size + (not drop_last and len(dataset) % batch_size)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[tuple[Tensor, list[dict[str, Tensor]]]]:
        indices = self._get_indices()
        languages = self._get_languages()

        for batch_idx in range(len(self)):
            start = batch_idx * self.batch_size
            end = start + self.batch_size

            batch_indices = indices[start::end]
            batch_language = languages[batch_idx]

            yield self._get_batch(batch_indices, batch_language)

    def set_language(self, language: str):
        self.language = language

    def _get_batch(self, indices: Tensor, language: str) -> tuple[Tensor, dict[str, Tensor]]:
        with ThreadPoolExecutor() as executor:
            images = torch.stack(list(executor.map(self.dataset.get_image, indices)))
        texts = self.dataset.get_texts(indices, language)
        return images, texts

    def _get_indices(self) -> Tensor:
        if self.shuffle:
            g = torch.Generator()
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))
        return indices

    def _get_languages(self) -> list[str]:
        if self.language is None:
            languages = [random.choice(["en", "ru"]) for _ in range(len(self))]
        else:
            languages = [self.language] * len(self)
        return languages
