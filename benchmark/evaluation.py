import argparse
from zeroshot_classification import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--dataset', type=str, default='all', help='Name of dataset')
    parser.add_argument('--model', type=str, default='google/siglip-base-patch16-224', help='Name of model')
    parser.add_argument('--task', type=str, default='zeroshot_classification', choices=['zeroshot_classification'],
                        help='Task')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='train or test part of dataset')
    parser.add_argument('--topk', type=int, default=1, help='k in topk accuracy')
    args = parser.parse_args()
    print(args.task, evaluate(args.model, dataset_name=args.dataset, k=args.topk))
