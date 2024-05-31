
# RuSigLIP

![Logo](images/Logo.png)

<p align="center">
  <a href="README_EN.md">English</a> | <a href="README.md">Русский</a>
</p>

## Overview
Russian language model for zero-shot image classification - implementation of the paper [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/ftp/arxiv/papers/2303/2303.15343.pdf) on PyTorch, executed as part of a spring project at HSE SPB in 2024.

## Installation
### Standard Installation via Pip
```sh
git clone https://github.com/Technolog796/RuSigLIP
pip install -r requirements.txt
python main.py
```

### Alternative Installation via uv
```sh
git clone https://github.com/Technolog796/RuSigLIP
uv venv venv --python=python3.X
source venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

## Using the Model

To start, use the command
```sh
accelerate launch train.py train_config.yml --accelerate_config configs/accelerate_config.yml
```

To evaluate the model, use the command
```sh
python benchmark/evaluation.py --dataset cifar100 --task zeroshot_classification --split test --size 100 --language en --topk 1 --model_weights path_to_model_weights
```
Evaluation metrics: accuracy, precision_macro, recall_macro, f1_macro

Datasets for evaluation: cifar10, cifar100, dtd, food101, oxfordiiitpet, mnist, country211, fgvcircraft, flowers102

To get predictions, use the command
```sh
python benchmark/predict.py --image image.jpg --labels "cat" "dog"
```

## Project Structure  
* <code>benchmark</code> - model evaluation
* <code>configs</code> - model and accelerate configurations
* <code>data_preprocessing</code> - data preparation: processing data from wikimedia, translation, translation evaluation, dataset evaluation (clipscore)  
* <code>dataset</code>  - working with data
* <code>experiments</code> - experiments that did not make it into the final version 
* <code>metrics</code> - jupyter notebooks with classification evaluation 
* <code>models</code> - main model files
* <code>telegram_bot</code> - telegram bot
* <code>test</code> - component testing
* <code>utils</code> - data loading, SigmoidLoss, auxiliary functions for training and inference

## Datasets
To train the model, we used the following datasets:

- [LAION COCO NLLB](https://huggingface.co/datasets/visheratin/laion-coco-nllb)
- [Conceptual 3M](https://ai.google.com/research/ConceptualCaptions/download)
- [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m)

For quick loading of these datasets, we recommend using the [img2dataset](https://github.com/rom1504/img2dataset) library.

## License
The repository is distributed under the [MIT](https://github.com/Technolog796/RuSigLIP/blob/main/LICENSE) license.
