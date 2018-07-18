FROM ubuntu:16.04

WORKDIR /bachelor-thesis-experiments
RUN apt-get update && apt-get install -y python3 python3-pip python3-tk
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD while :; do echo 'Hit CTRL+C'; sleep 1; done
# Run with python3 /bachelor-thesis-experiments/preprocessing-blogcatalog.py
