# FLIM+UNET script

# Usage

Create the docker image of this project by executing the following command at the root directory of this repository.

```sh
docker build -t unet-matheus .
```

Execute traning through docker image.

```sh
docker run --rm -it --ipc=host --gpus=all \
    -v <dataset path>:/app/BRATS2020 \
    -v <experiments path>:/app/exp \
    unet-matheus
```

where `<dataset path >` is the directory containing both the `trainning` and `validation` datasets and `<experiments path>` is the directory  to save the output images. 

These instructions assume that you have an NVIDIA gpu.
