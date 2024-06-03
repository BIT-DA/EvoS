
__all__ = ['configs_fmow_coral', 'configs_fmow_irm', 'configs_fmow_erm',
           'configs_fmow_erm_lisa', 'configs_fmow_erm_mixup', 'configs_fmow_agem',
           'configs_fmow_ewc', 'configs_fmow_ft', 'configs_fmow_si',
           'configs_fmow_simclr', 'configs_fmow_swav', "configs_fmow_drain",
           "configs_fmow_lssae", 'configs_fmow_cdot', 'configs_fmow_cida', 'configs_fmow_evos']

configs_fmow_erm =        {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/erm'}

configs_fmow_irm =        {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/irm'}

configs_fmow_coral =      {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/coral'}

configs_fmow_erm_mixup  = {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/erm_mixup'}

configs_fmow_erm_lisa =   {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'lisa': True, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/erm_lisa'}

configs_fmow_cdot =       {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'method': 'cdot', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/cdot'}

configs_fmow_cida =       {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'method': 'cida', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/cida'}

configs_fmow_lssae =      {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'lssae_coeff_y': 1.0, 'lssae_coeff_ts': 0.1, 'lssae_coeff_w': 1.0, 'lssae_zc_dim': 64, 'lssae_zw_dim': 64, 'method': 'lssae', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/lssae'}

configs_fmow_ft =         {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'K': 1, 'method': 'ft', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/ft'}

configs_fmow_simclr =     {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'method': 'simclr', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/simclr'}

configs_fmow_swav =       {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'method': 'swav', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/swav'}

configs_fmow_ewc =        {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/ewc'}

configs_fmow_si =         {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/si'}

configs_fmow_agem =       {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 1e-6, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/agem'}

configs_fmow_drain =      {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, "hidden_dim": 64, "latent_dim": 64, "num_rnn_layers": 10, "num_layer_to_replace": 1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/drain'}

configs_fmow_evos =       {'dataset': 'fmow', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 2e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 12, 'dim_head': 8, 'num_head': 64, 'scale': 3, 'truncate': 1.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 1024, 'warm_max_iters': None, 'warm_multiply': 10.0, 'method': 'evos', 'data_dir': '/data1/TL/data/wildtime/datasets/fMoW', 'log_dir': './checkpoints/fmow/evos'}
