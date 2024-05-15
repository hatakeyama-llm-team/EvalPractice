#!/bin/bash
export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/.venv_train/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train
echo "pyファイル: $1"
echo "データパス: $2"
echo "job名: $3"
python $1 $2 $3