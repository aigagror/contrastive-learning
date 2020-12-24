#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"
pip install tqdm
pip install matplotlib
pip install scikit-learn
python contrastive_learning.py
