# -*- coding:utf-8 -*-
# create: @time: 8/7/23 18:42

def set_config_for_extrapolation(config):
    # RoPE config
    config.use_rope_attention_bias = True
    config.rope_type = "mixed_base"  # "dynamic" of "linear" or "mixed_base"
    # when scale_factor=1.0, RoPE is NTKScaleRoPE, when scale_factor > 1, RoPE becomes DynamicallyNTKScaleRope
    config.rope_scaling_factor = 1.0
    config.fix_base = False  # please refer to https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/
    config.b = 0.6   # please refer to https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/

    #  alibi for encoder https://github.com/lucidrains/x-transformers/pull/88
    config.use_alibi = False
    config.learnable_alibi = False

    # attention scale
    config.use_entropy_scale = True  # https://openreview.net/forum?id=qc9O2EtrMI-

    # Others
    config.has_relative_attention_bias = False  # always set it to False

    config.consequent_visual_bias = True     # While applying RoPE on the visual component,
    # if True, the visual component is rotated from an angle starting from pos_idx = 0
    # if False, staring from a consecutive pos_idx after text component

    config.keep_visual_position_ids = True  # whether to keep visual pos_emb or not