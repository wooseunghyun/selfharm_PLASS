ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# 패키지 업데이트 및 필요한 패키지 설치
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

WORKDIR /selfharm_PLASS

# # Private git clone config
# RUN git config --global user.name "wooseunghyun"
# RUN git config --global user.email "shwoo10@swu.ac.kr"

RUN git clone https://github.com/wooseunghyun/selfharm_PLASS.git .

RUN pip install mmcv-full==1.7.1
RUN pip install -U openmim
RUN pip install seaborn
RUN mim install mmengine==0.8.2
RUN mim install mmdet==3.1.0
RUN mim install mmpose==1.1.0
RUN pip install moviepy==1.0.3
RUN pip install -e .

ENV FORCE_CUDA="0"

RUN apt-get update && apt-get install -y wget

RUN cd work_dirs/demo_rgbposec3d \
    &&  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg" -O pose_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt \
    &&  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h" -O rgb_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt 

# Set the default command to run when the container starts
CMD ["bash"]