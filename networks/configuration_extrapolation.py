# -*- coding:utf-8 -*-
# create: @time: 8/7/23 18:42

def set_config_for_extrapolation(config):
    config.use_rope_attention_bias = True
    config.rope_scaling_factor = 1.0
    config.rope_type = "dynamic"

    #  alibi for encoder https://github.com/lucidrains/x-transformers/pull/88
    config.use_alibi = False
    config.learnable_alibi = False

    config.use_entropy_scale = True  # https://openreview.net/forum?id=qc9O2EtrMI-

    # Others
    config.has_relative_attention_bias = False
    config.consequent_visual_bias = True