ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y gnupg2

RUN apt update && apt install -y git vim libglib2.0-0 libgl1-mesa-glx

WORKDIR /selfharm_PLASS

# # Private git clone config
# RUN git config --global user.name "wooseunghyun"
# RUN git config --global user.email "shwoo10@swu.ac.kr"

RUN git clone https://github.com/wooseunghyun/selfharm_PLASS.git .

RUN pip install mmcv-full
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install mmdet
RUN mim install mmpose
RUN pip install moviepy
RUN pip install -e .

RUN apt-get update && apt-get install -y wget

RUN cd work_dirs/demo_rgbposec3d \
    &&  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg" -O pose_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt \
    &&  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h" -O rgb_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt 

# Set the default command to run when the container starts
CMD ["bash"]