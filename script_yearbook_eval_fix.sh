#!/usr/bin/env bash


#===========eval-fix manner==========
# yearbook, incrementl-training methods
python3 main.py --dataset 'yearbook' --method 'ft' log_dir './checkpoints/yearbook/eval_fix/incrementl/IncFinetune' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'erm_mixup' log_dir './checkpoints/yearbook/eval_fix/incrementl/mixup' device 5 eval_fix True online_switch True
python3 main.py --dataset 'yearbook' --method 'simclr' log_dir './checkpoints/yearbook/eval_fix/incrementl/simclr' device 5 eval_fix True online_switch True
python3 main.py --dataset 'yearbook' --method 'swav' log_dir './checkpoints/yearbook/eval_fix/incrementl/swav' device 5 eval_fix True online_switch True
python3 main.py --dataset 'yearbook' --method 'ewc' log_dir './checkpoints/yearbook/eval_fix/incrementl/ewc' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'si' log_dir './checkpoints/yearbook/eval_fix/incrementl/si' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'agem' log_dir './checkpoints/yearbook/eval_fix/incrementl/agem' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'sgp' log_dir './checkpoints/yearbook/eval_fix/incrementl/sgp' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'drain' log_dir './checkpoints/yearbook/eval_fix/incrementl/drain' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'evos' log_dir './checkpoints/yearbook/eval_fix/incrementl/evos' device 5 eval_fix True


#yearbook, non-incrementl-training methods
python3 main.py --dataset 'yearbook' --method 'erm' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/offline' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'irm' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/irm' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'coral' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/coral' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'erm_mixup' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/mixup' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'erm_lisa' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/lisa' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'cdot' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/cdot' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'cida' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/cida' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'gi' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/gi' device 5 eval_fix True
python3 main.py --dataset 'yearbook' --method 'lssae' log_dir './checkpoints/yearbook/eval_fix/non-incrementl/lssae' device 5 eval_fix True


