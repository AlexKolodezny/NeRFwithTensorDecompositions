FROM nvidia/cuda:11.0-base

WORKDIR /code

COPY requirements.txt requirements.txt
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install python3-pip --yes && \
    pip3 install -r requirements.txt

CMD [ "bash" ]