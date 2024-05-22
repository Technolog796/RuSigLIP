import albumentations as A

from .dataset import RuSigLIPDataset


class WikiDataset(RuSigLIPDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )


class LaionCocoDataset(RuSigLIPDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    max_pixel_value=255.0,
                    always_apply=True,
                )
            ]
        )
