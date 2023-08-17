# Docker Setup

- [1. Requirements](#1-requirements)
- [2. Guide to use AnimeGanV2 in docker container](#2-guide-to-use-animeganv2-in-docker-container)
  - [2.1. Building docker image](#21-building-docker-image)
  - [2.2. Running AnimeGANv2 tools in container](#22-running-animeganv2-tools-in-container)

## 1. Requirements

Docker

## 2. Guide to use AnimeGanV2 in docker container

1. Clone this repo;
2. Build docker image;
3. Create folders for input/output files;
4. Run desired AnimeGANv2 tool in container using image you have built;

More details below.

### 2.1. Building docker image

Execute in terminal:

```shell
docker build --tag animegan:v2 .
```

[Dockerfile](./Dockerfile) will be used to build image that has proper python version and all requirements installed.

### 2.2. Running AnimeGANv2 tools in container

Step 1: start container shell

```shell
docker run --rm -it \
    --mount type=bind,source="$(pwd)"/dataset,target=/anime_gan_dir/dataset \
    --mount type=bind,source="$(pwd)"/results,target=/anime_gan_dir/results \
    --mount type=bind,source="$(pwd)"/video,target=/anime_gan_dir/video \
    animegan:v2
```

It will create container, mount folders `dataset`, `results`, and `video`, and connect to container's shell.

In terminal output, you must see something like that:

```shell
animegan@4f70da4ce086:/anime_gan_dir$ 
```

Then you can run scripts as if you would in repo working forlder. For exapmle, try to run test script:

```shell
python test.py --checkpoint_dir checkpoint/generator_Hayao_weight  --test_dir dataset/test/HR_photo --save_dir Hayao/HR_photo_docker
```

As a result, file [results/Hayao/HR_photo_docker/AE86.png](results/Hayao/HR_photo_docker/AE86.png) must appear

When you exit container started with `--rm`, it gets removed automatically. Image will persist.
