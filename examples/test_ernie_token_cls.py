# -*- coding:utf-8 -*-
# create: @time: 2/16/23 14:02
import torch
from networks import ErnieLayoutConfig, ErnieLayoutForTokenClassification, \
    ErnieLayoutImageProcessor, ErnieLayoutTokenizerFast, ERNIELayoutProcessor
from networks.model_util import ernie_cls_processing
from PIL import Image
import numpy as np


pretrain_torch_model_or_path = "path/to/pretrained/model"
doc_imag_path = "./dummy_input.jpeg"

device = torch.device("cuda:0")

def main():
    context = ['This is an example document', 'All ocr boxes are inserted into this list']
    layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # all boxes are resized between 0 - 1000
    labels = [1, 0]
    pil_image = Image.open(doc_imag_path).convert("RGB")

    # initialize tokenizer
    tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)

    # initialize feature extractor
    feature_extractor = ErnieLayoutImageProcessor(apply_ocr=False)
    processor = ERNIELayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    context_encodings = processor(pil_image, context)
    tokenized_res = ernie_cls_processing(context_encodings, layout, labels)

    # Tokenize context
    tokenized_res['input_ids'] = torch.tensor([tokenized_res['input_ids']]).to(device)
    tokenized_res['bbox'] = torch.tensor([tokenized_res['bbox']]).to(device)
    tokenized_res['labels'] = torch.tensor([tokenized_res['labels']]).to(device)
    tokenized_res['pixel_values'] = torch.tensor(np.array(context_encodings.data['pixel_values'])).to(device)

    # initialize config
    config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
    config.num_classes = 2  # num of classes

    # initialize ERNIE for TokenClassification
    model = ErnieLayoutForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=pretrain_torch_model_or_path,
        config=config,
    )
    model.to(device)

    output = model(**tokenized_res)

if __name__ == '__main__':
    main()