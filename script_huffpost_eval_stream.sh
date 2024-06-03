#!/usr/bin/env bash


#===========eval-stream manner==========
# huffpost, incrementl-training methods
python3 main.py --dataset 'huffpost' --method 'ft' log_dir './checkpoints/huffpost/eval_stream/IncFinetune' device 1 eval_fix False eval_next_timestamps 3
python3 main.py --dataset 'huffpost' --method 'erm_mixup' log_dir './checkpoints/huffpost/eval_stream/mixup' device 1 eval_fix False eval_next_timestamps 3 online_switch True
python3 main.py --dataset 'huffpost' --method 'ewc' log_dir './checkpoints/huffpost/eval_stream/ewc' device 1 eval_fix False eval_next_timestamps 3
python3 main.py --dataset 'huffpost' --method 'si' log_dir './checkpoints/huffpost/eval_stream/si' device 1 eval_fix False eval_next_timestamps 3
python3 main.py --dataset 'huffpost' --method 'agem' log_dir './checkpoints/huffpost/eval_stream/agem' device 1 eval_fix False eval_next_timestamps 3
python3 main.py --dataset 'huffpost' --method 'drain' log_dir './checkpoints/huffpost/eval_stream/drain' device 1 eval_fix False eval_next_timestamps 3
python3 main.py --dataset 'huffpost' --method 'evos' log_dir './checkpoints/huffpost/eval_stream/evos' device 1 eval_fix False eval_next_timestamps 3 warm_max_iters None warm_multiply 5.0 hidden_discriminator 128
