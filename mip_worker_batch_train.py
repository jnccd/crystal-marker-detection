import argparse
import os

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-c','--command', type=str, default='bash', help='.')
parser.add_argument('-n','--num-gpus', type=int, default='2', help='.')
parser.add_argument('-i','--image', type=str, default='ncarstensen/pcmd:0.13', help='.')
args = parser.parse_args()

for i in range(args.num_gpus):
    iter_command = f"screen -S w{i} -dm with_gpu -n 1 sudo mip-docker-run --rm --gpus '\"device=$CUDA_VISIBLE_DEVICES\"' {args.image} {args.command} -wi {i} -wc {args.num_gpus}"
    
    print(f"Starting run {i} of {args.num_gpus} with {iter_command}")
    os.system(iter_command)