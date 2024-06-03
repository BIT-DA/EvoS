
__all__ = ['configs_yearbook_coral', 'configs_yearbook_irm', 'configs_yearbook_erm',
           'configs_yearbook_erm_lisa', 'configs_yearbook_erm_mixup', 'configs_yearbook_agem',
           'configs_yearbook_ewc', 'configs_yearbook_ft', 'configs_yearbook_si',
           'configs_yearbook_simclr', 'configs_yearbook_swav', "configs_yearbook_drain",
           "configs_yearbook_lssae", 'configs_yearbook_cdot', 'configs_yearbook_cida', 'configs_yearbook_evos']

configs_yearbook_erm =        {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/erm'}

configs_yearbook_irm =        {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/irm'}

configs_yearbook_coral =      {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/coral'}

configs_yearbook_erm_mixup  = {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/erm_mixup'}

configs_yearbook_erm_lisa =   {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'lisa': True, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/erm_lisa'}

configs_yearbook_cdot =       {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'method': 'cdot', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/cdot'}

configs_yearbook_cida =       {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'method': 'cida', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/cida'}

configs_yearbook_gi =         {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'time_dim': 8, 'time_append_dim': 32, 'gi_finetune_bs': 64, 'gi_finetune_epochs': 50, 'gi_start_to_finetune': None, 'method': 'gi', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/gi'}

configs_yearbook_lssae =      {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'lssae_coeff_y': 1.0, 'lssae_coeff_ts': 0.1, 'lssae_coeff_w': 1.0, 'lssae_zc_dim': 32, 'lssae_zw_dim': 32, 'method': 'lssae', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/lssae'}

configs_yearbook_ft =         {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'K': 1, 'method': 'ft', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/ft'}

configs_yearbook_simclr =     {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'method': 'simclr', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/simclr'}

configs_yearbook_swav =       {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'method': 'swav', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/swav'}

configs_yearbook_ewc =        {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/ewc'}

configs_yearbook_si =         {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/si'}

configs_yearbook_agem =       {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/agem'}

configs_yearbook_drain =      {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, "hidden_dim": 32, "latent_dim": 32, "num_rnn_layers": 10, "num_layer_to_replace": -1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/drain'}

configs_yearbook_evos =       {'dataset': 'yearbook', 'yearbook_group_size': 4, 'device': 0, 'random_seed': 1, 'epochs': 50, 'lr': 0.001, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 15, 'dim_head': 8, 'num_head': 16, 'scale': 3, 'truncate': 1.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 64, 'warm_max_iters': 11000, 'warm_multiply': None, 'method': 'evos', 'data_dir': '/data1/TL/data/wildtime/datasets/yearbook', 'log_dir': './checkpoints/yearbook/evos'}
