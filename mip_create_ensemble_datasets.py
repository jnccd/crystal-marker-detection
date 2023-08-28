import argparse
import os

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-n','--name', type=str, default='test', help='.')
parser.add_argument('-mc','--max-config', type=int, default='1', help='.')
parser.add_argument('-sc','--step-config', type=int, default='0.1', help='.')
parser.add_argument('-np','--num-parts', type=int, default='5', help='.')
parser.add_argument('-op','--other-params', type=str, default='', help='.')
args = parser.parse_args()

for ic in range(0, args.max_config, args.step_config):
    for ip in range(args.num_parts):
        os.system(f'python traindata-creator/createDataset.py -n {args.name}-{ic}-p{ip} -a -aim 4 -agnc 1 {args.other_params} {ic} -taf /data/pcmd/dataset/_noise-ensemble-{args.name}')