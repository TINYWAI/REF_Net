vssm_mini_config = {'patch_size': 4, 'depths': [2, 2, 2, 2], 'dims': 96, 'ssm_d_state': 1, 'ssm_ratio': 2.0,
                    'ssm_rank_ratio': 2.0, 'ssm_dt_rank': 'auto', 'ssm_conv': 3, 'ssm_conv_bias': False,
                    'ssm_drop_rate': 0.0, 'ssm_init': 'v0', 'forward_type': 'v3noz', 'mlp_ratio': 4.0,
                    'mlp_drop_rate': 0.0, 'drop_path_rate': 0.2, 'patch_norm': True, 'downsample_version': 'v3',
                    'patchembed_version': 'v2', 'gmlp': False, 'use_checkpoint': False}

vssm_tiny_config = {'patch_size': 4, 'depths': [2, 2, 4, 2], 'dims': 96, 'ssm_d_state': 1, 'ssm_ratio': 2.0,
                    'ssm_rank_ratio': 2.0, 'ssm_dt_rank': 'auto', 'ssm_conv': 3, 'ssm_conv_bias': False,
                    'ssm_drop_rate': 0.0, 'ssm_init': 'v0', 'forward_type': 'v3noz', 'mlp_ratio': 4.0,
                    'mlp_drop_rate': 0.0, 'drop_path_rate': 0.2, 'patch_norm': True, 'downsample_version': 'v3',
                    'patchembed_version': 'v2', 'gmlp': False, 'use_checkpoint': False}

vssm_small_config = {'patch_size': 4, 'depths': [2, 2, 15, 2], 'dims': 96, 'ssm_d_state': 1, 'ssm_ratio': 2.0,
                     'ssm_rank_ratio': 2.0, 'ssm_dt_rank': 'auto', 'ssm_conv': 3, 'ssm_conv_bias': False,
                     'ssm_drop_rate': 0.0, 'ssm_init': 'v0', 'forward_type': 'v3noz', 'mlp_ratio': 4.0,
                     'mlp_drop_rate': 0.0, 'drop_path_rate': 0.3, 'patch_norm': True, 'downsample_version': 'v3',
                     'patchembed_version': 'v2', 'gmlp': False, 'use_checkpoint': False}

vssm_block_config = {'ssm_d_state': 1, 'ssm_ratio': 2.0, 'ssm_dt_rank': 'auto', 'ssm_conv': 3, 'ssm_conv_bias': False,
                     'ssm_drop_rate': 0.0, 'ssm_init': 'v0', 'forward_type': 'v3noz', 'mlp_ratio': 4.0,
                     'mlp_drop_rate': 0.0, 'gmlp': False, 'use_checkpoint': False}
