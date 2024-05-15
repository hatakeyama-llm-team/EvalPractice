#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train_backup
echo "pyファイル: $1"
echo "データパス: $2"
echo "job名: $3"
python $1 $2 $3



