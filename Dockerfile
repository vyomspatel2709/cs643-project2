FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    python3 \
    python3-pip \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV MODEL_PATH=/app/energy_model

RUN pip3 install --no-cache-dir pyspark==3.5.1 numpy

WORKDIR /app

COPY predict.py /app/predict.py
COPY energy_model /app/energy_model

ENTRYPOINT ["python3", "/app/predict.py"]
