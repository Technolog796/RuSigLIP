import argparse
from zeroshot_classification import evaluate
from datasets_get import all_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=all_datasets + ["all"],
        help="Name of dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="zeroshot_classification",
        choices=["zeroshot_classification"],
        help="Task",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="train or test part of dataset",
    )
    parser.add_argument(
        "--size", type=int, default=-1, help="Size of dataset. -1 for all dataset"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ru"],
        help="Language of dataset",
    )
    parser.add_argument("--topk", type=int, default=1, help="k in topk accuracy")
    parser.add_argument(
        "--model_weights",
        type=str,
        default="trained_models/model12/model.safetensors",
        help="Path to model weights",
    )
    args = parser.parse_args()
    if args.task == "zeroshot_classification":
        print(
            args.task,
            evaluate(
                model_weights=args.model_weights,
                dataset_name=args.dataset,
                split=args.split,
                size=args.size,
                language=args.language,
                k=args.topk,
            ),
        )
