# -*- coding:utf-8 -*-
# create: @time: 2/16/23 14:02
from PIL import Image
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

from networks import ErnieLayoutConfig, ErnieLayoutForTokenClassification, exErnieLayoutForTokenClassification, \
    ErnieLayoutTokenizerFast, ErnieLayoutProcessor, set_config_for_extrapolation

pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
doc_imag_path = "./dummy_input.jpeg"


def run_token_cls_with_ernie(model_type="ErnieLayoutForTokenClassification"):
    context = ['This is an example segment', 'All ocr boxes are inserted into this list']
    layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure all boxes are normalized between 0 - 1000
    labels = [1, 0]
    pil_image = Image.open(doc_imag_path).convert("RGB")

    # initialize tokenizer
    tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
    # setting tokenizer to pad on the right side of the sequence
    tokenizer.padding_side = 'right'

    # initialize feature extractor
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor = ErnieLayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    encoding = processor(pil_image, context, boxes=layout, word_labels=labels, return_tensors="pt")

    # initialize config
    config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
    config.num_classes = 2  # num of classes
    config.use_flash_attn = True  # use flash attention

    if model_type == "exErnieLayoutForTokenClassification":
        set_config_for_extrapolation(config)

    # initialize ERNIE for TokenClassification
    model = eval(model_type).from_pretrained(
        pretrained_model_name_or_path=pretrain_torch_model_or_path,
        config=config,
    )
    outputs = model(**encoding)
    loss = outputs.loss
    logits = outputs.logits


if __name__ == '__main__':
    model_type = "ErnieLayoutForTokenClassification" # "exErnieLayoutForTokenClassification"
    run_token_cls_with_ernie(model_type=model_type)
