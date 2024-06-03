
__all__ = ['configs_rmnist_coral', 'configs_rmnist_irm', 'configs_rmnist_erm',
           'configs_rmnist_erm_lisa', 'configs_rmnist_erm_mixup', 'configs_rmnist_agem',
           'configs_rmnist_ewc', 'configs_rmnist_ft', 'configs_rmnist_si',
           'configs_rmnist_simclr', 'configs_rmnist_swav', "configs_rmnist_drain",
           "configs_rmnist_lssae", 'configs_rmnist_cdot', 'configs_rmnist_cida', 'configs_rmnist_evos']

configs_rmnist_erm =        {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/erm'}

configs_rmnist_irm =        {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/irm'}

configs_rmnist_coral =      {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/coral'}

configs_rmnist_erm_mixup  = {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/erm_mixup'}

configs_rmnist_erm_lisa =   {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'lisa': True, 'mix_alpha': 1.0, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/erm_lisa'}

configs_rmnist_cdot =       {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'method': 'cdot', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/cdot'}

configs_rmnist_cida =       {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'method': 'cida', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/cida'}

configs_rmnist_gi =         {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'time_dim': 8, 'time_append_dim': 32, 'gi_finetune_bs': 64, 'gi_finetune_epochs': 50, 'gi_start_to_finetune': None, 'method': 'gi', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/gi'}

configs_rmnist_lssae =      {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'lssae_coeff_y': 1.0, 'lssae_coeff_ts': 0.1, 'lssae_coeff_w': 0.1, 'lssae_zc_dim': 64, 'lssae_zw_dim': 64, 'method': 'lssae', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/lssae'}

configs_rmnist_ft =         {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'K': 1, 'method': 'ft', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/ft'}

configs_rmnist_simclr =     {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'method': 'simclr', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/simclr'}

configs_rmnist_swav =       {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'method': 'swav', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/swav'}

configs_rmnist_ewc =        {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/ewc'}

configs_rmnist_si =         {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/si'}

configs_rmnist_agem =       {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-5, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/agem'}

configs_rmnist_drain =      {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, "hidden_dim": 64, "latent_dim": 64, "num_rnn_layers": 10, "num_layer_to_replace": -1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/drain'}

configs_rmnist_evos =       {'dataset': 'rmnist', 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 1e-3, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 5, 'dim_head': 8, 'num_head': 32, 'scale': 3, 'truncate': 2.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 256, 'warm_max_iters': None, 'warm_multiply': 2.0, 'method': 'evos', 'data_dir': '/data1/TL/data/wildtime/datasets/rmnist', 'log_dir': './checkpoints/rmnist/evos'}
