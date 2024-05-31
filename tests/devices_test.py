import torch
from accelerate import Accelerator


def check_cuda_availability() -> None:
    """
    This function checks the availability of CUDA-enabled GPUs and the number of such devices.
    It prints whether CUDA is available and the count of CUDA devices.

    Returns:
        None
    """
    is_cuda_available: bool = torch.cuda.is_available()
    cuda_device_count: int = torch.cuda.device_count()

    print(f"CUDA available: {is_cuda_available}")
    print(f"Number of CUDA devices: {cuda_device_count}")


def test_accelerate() -> None:
    """
    This function tests the 'accelerate' library by initializing an Accelerator instance.
    It checks if the Accelerator can correctly detect the available CUDA devices.

    Returns:
        None
    """
    accelerator: Accelerator = Accelerator()
    device: torch.device = accelerator.device
    print(f"Accelerate is using device: {device}")


if __name__ == "__main__":
    check_cuda_availability()
    test_accelerate()
