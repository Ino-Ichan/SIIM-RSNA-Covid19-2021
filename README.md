# 

## environment

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


