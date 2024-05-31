# RuSigLIP

## Обзор
Русскоязычная модель для zero-shot классификации изображений - реализация статьи [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/ftp/arxiv/papers/2303/2303.15343.pdf) на PyTorch, выполненная в рамках весеннего проекта в НИУ ВШЭ СПБ в 2024 году.

## Установка
### Стандартная установка через Pip
```sh
git clone https://github.com/Technolog796/RuSigLIP
pip install -r requirements.txt
python main.py
```

### Альтернативная установка через uv
```sh
git clone https://github.com/Technolog796/RuSigLIP
uv venv venv --python=python3.X
source venv/bin/activate
uv pip install -r requirements.txt
python main.py
```


## Использование модели

Для запуска воспользуйтесь командой
```sh
accelerate launch train.py train_config.yml --accelerate_config configs/accelerate_config.yml
```

Для оценки модели воспользуйтесь командой

```sh
python benchmark/evaluation.py --dataset cifar100 --task zeroshot_classification --split test --size 100 --language en --topk 1
```
Метрики для оценки: accuracy, precision_macro, recall_macro, f1_macro\
Датасеты для оценки: cifar10, cifar100, dtd, food101, oxfordiiitpet, mnist, country211, fgvcircraft, flowers102  \

Для получения предсказаний воспользуйтесь командой
```sh
python benchmark/predict.py --image image.jpg --labels "cat" "dog"
```

## Структура проекта  
* <code>benchmark</code> - оценка модели
* <code>configs</code> - конфиги модели и accelerate
* <code>data</code> - подготовка данных: обработка данных с wikimedia, перевод, оценка перевода, оценка датасета (clipscore)  
* <code>dataset</code>  - работа с данными
* <code>experiments</code> - эксперименты, которые не попали в финальную версию 
* <code>metrics</code> - jupyter notebooks с оценкой классификации 
* <code>models</code> - основные файлы модели
* <code>telegram_bot</code> - телеграм бот
* <code>test</code> - проверка работоспособности компонентов
* <code>utils</code> - загрузка данных, SigmoidLoss, вспомогательные функции для train и inference


