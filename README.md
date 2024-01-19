# 유치장 내 자살자해 탐지 PLASS

이 저장소는 유치장 내 자살자해 탐지의 구현을 포함하고 있습니다. 이 프로젝트는 영상, 스켈레톤 기반 이상행동 탐지를 위해 MMAction2 프레임워크를 활용합니다.

## 모듈 사용방법

프로젝트를 실행 방법


### 1. 환경설정

```
  conda create -n selfharm python==3.8
  conda activate selfharm
  git clone https://github.com/wooseunghyun/selfharm_PLASS.git .
  pip install -U openmim
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install chardet
  mim install mmcv-full
  mim install mmengine
  mim install mmdet
  mim install mmpose
  pip install moviepy
  pip install -e .
```

### 2. pth 파일 다운로드

```
  cd work_dirs/demo_rgbposec3d
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hfvHrenziejJyLPlHvpttWBJB_GeVlRg" -O pose_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h" -O rgb_best_acc_top1_epoch_17.pth && rm -rf ~/cookies.txt
```

또는

pose의 pth파일 [GoogleDrive](https://drive.google.com/file/d/1hfvHrenziejJyLPlHvpttWBJB_GeVlRg/view?usp=sharing)


rgb의 pth파일 [GoogleDrive](https://drive.google.com/file/d/1aI0K3M5r1K4tk2x9-s5hsbdPE8miW6_h/view?usp=sharing)


### 3. 모듈 실행

```
  python ./demo/module_video_structuralize_rgbposec3d.py
```

또는 

module_video_structuralize_rgbposec3d.py 내의 selfharm_detection()을 사용하여 결과를 얻을 수 있음.


단, 908라인에 detect_m() 대신 적절한 데이터를 입력해야함.


받아와야하는 데이터는 modified_results, person_bboxes, pose_results, frames
