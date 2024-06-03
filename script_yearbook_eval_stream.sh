#!/usr/bin/env bash


#===========eval-stream manner==========
# yearbook, incrementl-training methods
python3 main.py --dataset 'yearbook' --method 'ft' log_dir './checkpoints/yearbook/eval_stream/IncFinetune' device 6 eval_fix False eval_next_timestamps 5
python3 main.py --dataset 'yearbook' --method 'erm_mixup' log_dir './checkpoints/yearbook/eval_stream/mixup' device 6 eval_fix False eval_next_timestamps 5 online_switch True
python3 main.py --dataset 'yearbook' --method 'simclr' log_dir './checkpoints/yearbook/eval_stream/simclr' device 6 eval_fix False eval_next_timestamps 5 online_switch True
python3 main.py --dataset 'yearbook' --method 'swav' log_dir './checkpoints/yearbook/eval_stream/swav' device 6 eval_fix False eval_next_timestamps 5 online_switch True
python3 main.py --dataset 'yearbook' --method 'ewc' log_dir './checkpoints/yearbook/eval_stream/ewc' device 6 eval_fix False eval_next_timestamps 5
python3 main.py --dataset 'yearbook' --method 'si' log_dir './checkpoints/yearbook/eval_stream/si' device 6 eval_fix False eval_next_timestamps 5
python3 main.py --dataset 'yearbook' --method 'agem' log_dir './checkpoints/yearbook/eval_stream/agem' device 6 eval_fix False eval_next_timestamps 5
python3 main.py --dataset 'yearbook' --method 'drain' log_dir './checkpoints/yearbook/eval_stream/drain' device 6 eval_fix False eval_next_timestamps 5
python3 main.py --dataset 'yearbook' --method 'evos' log_dir './checkpoints/yearbook/eval_stream/evos' device 6 eval_fix False eval_next_timestamps 5 hidden_discriminator 256 warm_max_iters None warm_multiply 30.0
