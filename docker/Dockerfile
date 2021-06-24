#FROM pytorch/pytorch:latest
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git tmux libopencv-dev

RUN pip install tensorboard opencv-python
RUN conda install -y jupyter jupyterlab
RUN conda install -y pandas scikit-learn matplotlib seaborn
RUN pip install albumentations==0.5.2
RUN pip install segmentation-models-pytorch==0.1.3
RUN pip install timm==0.4.9
RUN pip install pytorch-gradcam==0.2.1
RUN pip install pydicom==2.1.2 pylibjpeg==1.3.0 pylibjpeg-libjpeg==1.2.0
RUN pip install numpy==1.20.0
RUN pip install neptune-client==0.9.16
RUN pip install iterative-stratification==0.1.6
RUN pip install wandb==0.10.31