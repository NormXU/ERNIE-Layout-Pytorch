import unittest
import os, sys
import torch
from PIL import Image
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor
from networks import ErnieLayoutConfig, ErnieLayoutForTokenClassification, exErnieLayoutForTokenClassification, \
    ErnieLayoutTokenizerFast, ErnieLayoutProcessor, set_config_for_extrapolation

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_PATH)


def prepare_dummy_encodings(pretrain_torch_model_or_path, doc_imag_path):
    context = ['This is an example segment', 'All ocr boxes are inserted into this list']
    layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure all boxes are normalized between 0 - 1000
    labels = [1, 0]
    pil_image = Image.open(doc_imag_path).convert("RGB")

    # initialize tokenizer
    tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)

    # initialize feature extractor
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor = ErnieLayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
    return processor(pil_image, context, boxes=layout, word_labels=labels, return_tensors="pt")


class TestFlashAttn(unittest.TestCase):
    def setUp(self) -> None:
        doc_imag_path = os.path.join(PROJECT_ROOT_PATH, "examples/dummy_input.jpeg")
        pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
        self.encoding = prepare_dummy_encodings(pretrain_torch_model_or_path, doc_imag_path)
        self.pretrain_model_pth = pretrain_torch_model_or_path

    def test_vanilla_ernie_layout(self):
        # initialize config
        config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=self.pretrain_model_pth)
        config.num_classes = 2  # num of classes
        # with flash attention
        config.use_flash_attn = True

        # initialize ERNIE for TokenClassification
        model_flash = ErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_w_fa = model_flash(**self.encoding).hidden_states

        # without flash attention
        config.use_flash_attn = False

        # initialize ERNIE for TokenClassification
        model_ = ErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_wo_fa = model_(**self.encoding).hidden_states
        assert torch.mean(hs_wo_fa - hs_w_fa) < 1e-8

    def test_ex_ernie_layout_rope(self):
        # initialize config
        config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=self.pretrain_model_pth)
        config.num_classes = 2  # num of classes
        set_config_for_extrapolation(config)
        # with flash attention
        config.use_flash_attn = True

        # initialize ERNIE for TokenClassification
        model_flash = exErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_w_fa = model_flash(**self.encoding).hidden_states

        # without flash attention
        config.use_flash_attn = False

        # initialize ERNIE for TokenClassification
        model_ = exErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_wo_fa = model_(**self.encoding).hidden_states
        assert torch.mean(hs_wo_fa - hs_w_fa) < 1e-8

    def test_ex_ernie_layout_alibi(self):
        # initialize config
        config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=self.pretrain_model_pth)
        config.num_classes = 2  # num of classes
        set_config_for_extrapolation(config)
        config.use_rope_attention_bias = False
        config.use_alibi = True
        # with flash attention
        config.use_flash_attn = True

        # initialize ERNIE for TokenClassification
        model_flash = exErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_w_fa = model_flash(**self.encoding).hidden_states

        # without flash attention
        config.use_flash_attn = False

        # initialize ERNIE for TokenClassification
        model_ = exErnieLayoutForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrain_model_pth,
            config=config,
        ).eval()
        hs_wo_fa = model_(**self.encoding).hidden_states
        assert torch.mean(hs_wo_fa - hs_w_fa) < 1e-8


if __name__ == '__main__':
    unittest.main()
