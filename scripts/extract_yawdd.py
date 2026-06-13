#!/usr/bin/env python3
import gzip
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--dst', default=os.path.join('data','raw','YawDD.rar'))
args = parser.parse_args()

os.makedirs(os.path.dirname(args.dst), exist_ok=True)
with gzip.open(args.src, 'rb') as f_in:
    with open(args.dst, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print('wrote', args.dst)
