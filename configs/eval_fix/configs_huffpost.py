
__all__ = ['configs_huffpost_coral', 'configs_huffpost_irm', 'configs_huffpost_erm',
           'configs_huffpost_erm_lisa', 'configs_huffpost_erm_mixup', 'configs_huffpost_agem',
           'configs_huffpost_ewc', 'configs_huffpost_ft', 'configs_huffpost_si',
            'configs_huffpost_drain', 'configs_huffpost_evos']

configs_huffpost_erm =        {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/erm'}

configs_huffpost_irm =        {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/irm'}

configs_huffpost_coral =      {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/coral'}

configs_huffpost_erm_mixup  = {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/erm_mixup'}

configs_huffpost_erm_lisa =   {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'lisa': True, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/erm_lisa'}

configs_huffpost_gi =         {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'time_dim': 8, 'time_append_dim': 32, 'gi_finetune_bs': 64, 'gi_finetune_epochs': 50, 'gi_start_to_finetune': None, 'method': 'gi', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/gi'}

configs_huffpost_ft =         {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'K': 1, 'method': 'ft', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/ft'}

configs_huffpost_ewc =        {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/ewc'}

configs_huffpost_si =         {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/si'}

configs_huffpost_agem =       {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-7, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/agem'}

configs_huffpost_drain =      {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, "hidden_dim": 64, "latent_dim": 64, "num_rnn_layers": 10, "num_layer_to_replace": 1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/drain'}

configs_huffpost_evos =       {'dataset': 'huffpost', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2012, 'split_time': 2015, 'dim_head': 8, 'num_head': 128, 'scale': 3, 'truncate': 1.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 1024, 'warm_max_iters': 1000, 'warm_multiply': None, 'method': 'evos', 'data_dir': '/data1/TL/data/wildtime/datasets/huffpost', 'log_dir': './checkpoints/huffpost/evos'}
