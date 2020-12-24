#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"
python main.py --bsz=1024 --epochs=1 --method=supcon --lr=1e-3
