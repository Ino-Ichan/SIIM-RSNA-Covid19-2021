# 

## environment

#### Clone repository

```
git clone --recursive git@github.com:Ino-Ichan/SIIM-RSNA-Covid19-2021.git
```





# Study level and Image level none
#### Docker build
```
docker build -t siim-rsna-2021:study_v0 -f docker/Dockerfile docker/
```

#### Docker run
```
docker run -it --rm --name siim_rsna_study\
 --gpus all --shm-size=100g\
 -v $PWD:/workspace\
 -p 1111:8888 -p 1001:6006 --ip=host\
 siim-rsna-2021:study_v0
```

# Create directory

Please create `./data`, `./output` directory 


# yolov5

#### Docker build
```
cd customized_yolov5
docker build -t siim-rsna-2021:yolov5 -f Dockerfile ./
cd ..
```

#### Docker run
```
docker run -it --rm --name siim_rsna_yolov5\
 --gpus all --shm-size=100g\
 -v $PWD:/workspace\
 -p 5555:8888 -p 5005:6006 --ip=host\
 siim-rsna-2021:yolov5
```

# mmdet

### Docker build

pytorch version 1.7.0
```
docker build -t siim_mmdet:v0 -f docker/Dockerfile_mmdet docker/
```
### Docker run

##### RTX3090

```
docker run -it --rm --name siim_mmdetection\
 --gpus '"device=1"' --shm-size=100g\
 -v $PWD:/workspace\
 -p 7777:8888 -p 7007:6006 --ip=host\
 siim_mmdet:v0
```

##### 0:2080Ti 1:RTX3090

```
docker run -it --rm --name siim_mmdetection\
 --gpus '"device=0,1"' --shm-size=100g\
 -v $PWD:/workspace\
 -p 7777:8888 -p 7007:6006 --ip=host\
 siim_mmdet:v0
```