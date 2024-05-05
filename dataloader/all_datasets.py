import torch
import torchvision.transforms.v2 as transforms

from .dataset import RuSigLIPDataset


class WikiDataset(RuSigLIPDataset):
    def __init__(self, target_image_size=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((target_image_size, target_image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )


class LaionCocoDataset(RuSigLIPDataset):
    def __init__(self, target_image_size=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((target_image_size, target_image_size)),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]
        )
