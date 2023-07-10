# -*- coding:utf-8 -*-
# create: @time: 2/16/23 14:02
from PIL import Image
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

from networks import ErnieLayoutConfig, ErnieLayoutForTokenClassification, ErnieLayoutTokenizerFast, \
    ErnieLayoutProcessor

pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
doc_imag_path = "./dummy_input.jpeg"


def main():
    context = ['This is an example segment', 'All ocr boxes are inserted into this list']
    layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure all boxes are normalized between 0 - 1000
    labels = [1, 0]
    pil_image = Image.open(doc_imag_path).convert("RGB")

    # initialize tokenizer
    tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)

    # initialize feature extractor
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor = ErnieLayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    encoding = processor(pil_image, context, boxes=layout, word_labels=labels, return_tensors="pt")

    # initialize config
    config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
    config.num_classes = 2  # num of classes

    # initialize ERNIE for TokenClassification
    model = ErnieLayoutForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=pretrain_torch_model_or_path,
        config=config,
    )

    outputs = model(**encoding)
    loss = outputs.loss
    logits = outputs.logits


if __name__ == '__main__':
    main()
