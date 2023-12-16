import argparse
import os

parser = argparse.ArgumentParser(prog='mip-worker-batch-train', description='Wraps multiple batch train runs in screen sessions and docker containers and automatically passes the worker arguments into the batch train instances for automated multi gpu ensemble training.')
parser.add_argument('-c','--command', type=str, default='bash', help='The batch train training command.')
parser.add_argument('-n','--num-gpus', type=int, default='2', help='The number of gpus that should be used.')
parser.add_argument('-i','--image', type=str, default='ncarstensen/pcmd:0.13', help='The image to use for the docker container.')
args = parser.parse_args()

for i in range(args.num_gpus):
    iter_command = f"screen -S w{i} -dm with_gpu -n 1 sudo mip-docker-run --rm --gpus '\"device=$CUDA_VISIBLE_DEVICES\"' {args.image} {args.command} -wi {i} -wc {args.num_gpus}"
    
    print(f"Starting run {i} of {args.num_gpus} with {iter_command}")
    os.system(iter_command)