#!/usr/bin/env bash


#===========eval-fix manner==========
# huffpost, incrementl-training methods
python3 main.py --dataset 'huffpost' --method 'ft' log_dir './checkpoints/huffpost/eval_fix/incrementl/IncFinetune' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'erm_mixup' log_dir './checkpoints/huffpost/eval_fix/incrementl/mixup' device 2 eval_fix True online_switch True
python3 main.py --dataset 'huffpost' --method 'ewc' log_dir './checkpoints/huffpost/eval_fix/incrementl/ewc' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'si' log_dir './checkpoints/huffpost/eval_fix/incrementl/si' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'agem' log_dir './checkpoints/huffpost/eval_fix/incrementl/agem' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'drain' log_dir './checkpoints/huffpost/eval_fix/incrementl/drain' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'evos' log_dir './checkpoints/huffpost/eval_fix/incrementl/evos' device 2 eval_fix True


#huffpost, non-incrementl-training methods
python3 main.py --dataset 'huffpost' --method 'erm' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/offline' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'irm' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/irm' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'coral' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/coral' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'erm_mixup' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/mixup' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'erm_lisa' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/lisa' device 2 eval_fix True
python3 main.py --dataset 'huffpost' --method 'gi' log_dir './checkpoints/huffpost/eval_fix/non-incrementl/gi' device 2 eval_fix True


