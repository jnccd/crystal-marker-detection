#!/usr/bin/env bash

# create docker container with GPUs on MIP-Server (lena)
# -c | --command:   command to execute
# -n | --num-gpus:  gpu count
# -i | --image:     docker image to use

# base cases
N=2
COMMAND=bash
IMAGE=ncarstensen/pcmd:0.1

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--command)
            COMMAND=$2
            shift 2
            ;;
        -n|--num-workers)
            N=$2
            shift 2
            ;;
        -i|--image)
            IMAGE=$2
            shift 2
            ;;
        -*|--*)
            echo "Unknown option $1"
            ;;
        *)
            shift
            ;;
    esac
done

for (( i = 0; i < $N; i++ )); do 
    with_gpu -n 1 sudo mip-docker-run --rm --gpus '"device=$CUDA_VISIBLE_DEVICES"' $IMAGE $COMMAND -wi $i -wc $N &
done
