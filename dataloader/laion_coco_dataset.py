import albumentations as A

from .dataset import RuSigLIPDataset


class LaionCocoDataset(RuSigLIPDataset):
    def __init__(self, data_file, tokenizer, target_size=256, max_len=64) -> None:
        super().__init__(data_file, tokenizer, target_size, max_len)

        self.transforms = A.Compose(
            [
                A.Resize(self.target_size, self.target_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )

