FROM nvidia/cuda:11.7.0-base-ubuntu20.04

WORKDIR /code

COPY requirements.txt requirements.txt
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install python3-pip --yes && \
    pip3 install -r requirements.txt

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt install  openssh-server sudo -y

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 akolodeznyi

RUN  echo 'akolodeznyi:123456' | chpasswd

RUN service ssh start

CMD ["/usr/sbin/sshd","-D"]