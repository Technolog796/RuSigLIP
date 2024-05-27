import albumentations as A

from .dataset import RuSigLIPDataset


class WikiDataset(RuSigLIPDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
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
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    max_pixel_value=255.0,
                    always_apply=True,
                )
            ]
        )


class CIFAR10(LaionCocoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                    max_pixel_value=255.0,
                    always_apply=True,
                )
            ]
        )


class CIFAR100(LaionCocoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.50707516, 0.48654887, 0.44091784],
                    std=[0.26733429, 0.25643846, 0.27615047],
                    max_pixel_value=255.0,
                    always_apply=True,
                )
            ]
        )


class MNIST(LaionCocoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
