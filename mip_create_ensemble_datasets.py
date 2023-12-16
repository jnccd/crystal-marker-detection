import argparse
import os
import sys

parser = argparse.ArgumentParser(prog='mip-create-ensemble-datasets', description='Creates ensemble datasets for the mip server.')
parser.add_argument('-n','--name', type=str, default='test', help='Name of the datasets.')
parser.add_argument('-mc','--max-config', type=float, default='1', help='The maximum value of the config range.')
parser.add_argument('-sc','--step-config', type=float, default='0.1', help='The step size of the config range.')
parser.add_argument('-np','--num-parts', type=int, default='5', help='The number of datasets per config.')
parser.add_argument('-op','--other-params', type=str, default='', help='These will be passed through to createDataset.')
args = parser.parse_args()

def float_range(start, stop, step):
    nums = []
    i = start
    while i < stop:
        nums.append(i)
        i += step
    nums.append(stop)
    return [round(x, 10) for x in nums]

for ic in float_range(0, args.max_config, args.step_config):
    for ip in range(args.num_parts):
        os.system(f'python traindata-creator/createDataset.py -n {args.name}-{ic}-p{ip} -a -aim 4 -agnc 1 {args.other_params} {ic} -taf /data/pcmd/dataset/_noise-ensemble-{args.name}')