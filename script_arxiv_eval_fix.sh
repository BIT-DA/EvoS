#!/usr/bin/env bash


#===========eval-fix manner==========
# arxiv, incrementl-training methods
python3 main.py --dataset 'arxiv' --method 'ft' log_dir './checkpoints/arxiv/eval_fix/incrementl/IncFinetune' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'erm_mixup' log_dir './checkpoints/arxiv/eval_fix/incrementl/mixup' device 3 eval_fix True online_switch True
python3 main.py --dataset 'arxiv' --method 'ewc' log_dir './checkpoints/arxiv/eval_fix/incrementl/ewc' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'si' log_dir './checkpoints/arxiv/eval_fix/incrementl/si' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'agem' log_dir './checkpoints/arxiv/eval_fix/incrementl/agem' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'drain' log_dir './checkpoints/arxiv/eval_fix/incrementl/drain' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'evos' log_dir './checkpoints/arxiv/eval_fix/incrementl/evos' device 3 eval_fix True


#arxiv, non-incrementl-training methods
python3 main.py --dataset 'arxiv' --method 'erm' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/offline' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'irm' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/irm' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'coral' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/coral' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'erm_mixup' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/mixup' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'erm_lisa' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/lisa' device 3 eval_fix True
python3 main.py --dataset 'arxiv' --method 'gi' log_dir './checkpoints/arxiv/eval_fix/non-incrementl/gi' device 3 eval_fix True


