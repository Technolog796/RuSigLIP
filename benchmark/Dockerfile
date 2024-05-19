FROM python:3.10

WORKDIR /benchmark

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt



COPY evaluation.py evaluation.py
COPY datasets.py datasets.py
COPY zeroshot_classification.py zeroshot_classification.py
COPY predict.py predict.py