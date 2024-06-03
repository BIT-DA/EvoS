#!/usr/bin/env bash


#===========eval-fix manner==========
# fmow, incrementl-training methods
python3 main.py --dataset 'fmow' --method 'ft' log_dir './checkpoints/fmow/eval_fix/incrementl/IncFinetune' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'erm_mixup' log_dir './checkpoints/fmow/eval_fix/incrementl/mixup' device 6 eval_fix True online_switch True lr 2e-5
python3 main.py --dataset 'fmow' --method 'simclr' log_dir './checkpoints/fmow/eval_fix/incrementl/simclr' device 6 eval_fix True online_switch True
python3 main.py --dataset 'fmow' --method 'swav' log_dir './checkpoints/fmow/eval_fix/incrementl/swav' device 6 eval_fix True online_switch True
python3 main.py --dataset 'fmow' --method 'ewc' log_dir './checkpoints/fmow/eval_fix/incrementl/ewc' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'si' log_dir './checkpoints/fmow/eval_fix/incrementl/si' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'agem' log_dir './checkpoints/fmow/eval_fix/incrementl/agem' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'drain' log_dir './checkpoints/fmow/eval_fix/incrementl/drain' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'evos' log_dir './checkpoints/fmow/eval_fix/incrementl/evos' device 6 eval_fix True


#fmow, non-incrementl-training methods
python3 main.py --dataset 'fmow' --method 'erm' log_dir './checkpoints/fmow/eval_fix/non-incrementl/offline' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'irm' log_dir './checkpoints/fmow/eval_fix/non-incrementl/irm' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'coral' log_dir './checkpoints/fmow/eval_fix/non-incrementl/coral' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'erm_mixup' log_dir './checkpoints/fmow/eval_fix/non-incrementl/mixup' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'erm_lisa' log_dir './checkpoints/fmow/eval_fix/non-incrementl/lisa' device 6 eval_fix True
python3 main.py --dataset 'fmow' --method 'lssae' log_dir './checkpoints/fmow/eval_fix/non-incrementl/lssae' device 6 eval_fix True


