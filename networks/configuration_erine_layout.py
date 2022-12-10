# coding=utf-8
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ErnieLayoutConfig(PretrainedConfig):
    model_type = "ernie_layout"

    def __init__(
            self,
            vocab_size=250002,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=100,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            max_2d_position_embeddings=1024,
            coordinate_size=128,
            shape_size=128,
            has_relative_attention_bias=True,
            rel_pos_bins=32,
            max_rel_pos=128,
            rel_2d_pos_bins=64,
            max_rel_2d_pos=256,
            has_spatial_attention_bias=True,
            input_size=224,
            classifier_dropout=None,
            gradient_checkpointing=False,
            has_visual_segment_embedding=False,
            image_feature_pool_shape=[7, 7, 256],
            output_past=True,
            **kwargs
    ):
        """Constructs RobertaConfig."""
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.classifier_dropout = classifier_dropout
        self.coordinate_size = coordinate_size
        self.gradient_checkpointing = gradient_checkpointing
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.input_size = input_size
        self.image_feature_pool_shape = image_feature_pool_shape
        self.max_rel_pos = max_rel_pos
        self.max_rel_2d_pos = max_rel_2d_pos
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.output_past = output_past
        self.rel_pos_bins = rel_pos_bins
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.shape_size = shape_size
