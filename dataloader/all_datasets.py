import albumentations as A

from .dataset import RuSigLIPDataset


class WikiDataset(RuSigLIPDataset):
    def __init__(self, target_image_size=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = A.Compose(
            [
                A.Resize(target_image_size, target_image_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )


class LaionCocoDataset(RuSigLIPDataset):
    def __init__(self, target_image_size=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = A.Compose(
            [
                A.Resize(target_image_size, target_image_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )
