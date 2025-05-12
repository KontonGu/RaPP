FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --disable-pip-version-check -U -r requirements.txt
COPY server server
