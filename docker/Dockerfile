FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1

RUN pip install \
        torch-geometric==2.5.3 \
        numpy \
        scipy \
        texttable \
        tqdm \
        dill \
        pandas \
        seaborn \
        matplotlib

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "train_smfe_hgt.py", "--epochs", "10", "--batches-per-epoch", "20"]
