#!/usr/bin/env bash


#===========eval-fix manner==========
# rmnist, incrementl-training methods
python3 main.py --dataset 'rmnist' --method 'ft' log_dir './checkpoints/rmnist/eval_fix/incrementl/IncFinetune' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'erm_mixup' log_dir './checkpoints/rmnist/eval_fix/incrementl/mixup' device 5 eval_fix True online_switch True
python3 main.py --dataset 'rmnist' --method 'simclr' log_dir './checkpoints/rmnist/eval_fix/incrementl/simclr' device 5 eval_fix True online_switch True
python3 main.py --dataset 'rmnist' --method 'swav' log_dir './checkpoints/rmnist/eval_fix/incrementl/swav' device 5 eval_fix True online_switch True
python3 main.py --dataset 'rmnist' --method 'ewc' log_dir './checkpoints/rmnist/eval_fix/incrementl/ewc' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'si' log_dir './checkpoints/rmnist/eval_fix/incrementl/si' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'agem' log_dir './checkpoints/rmnist/eval_fix/incrementl/agem' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'drain' log_dir './checkpoints/rmnist/eval_fix/incrementl/drain' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'evos' log_dir './checkpoints/rmnist/eval_fix/incrementl/evos' device 5 eval_fix True


#rmnist, non-incrementl-training methods
python3 main.py --dataset 'rmnist' --method 'erm' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/offline' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'irm' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/irm' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'coral' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/coral' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'erm_mixup' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/mixup' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'erm_lisa' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/lisa' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'cdot' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/cdot' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'cida' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/cida' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'gi' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/gi' device 5 eval_fix True
python3 main.py --dataset 'rmnist' --method 'lssae' log_dir './checkpoints/rmnist/eval_fix/non-incrementl/lssae' device 5 eval_fix True


