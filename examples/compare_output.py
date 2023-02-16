# -*- coding:utf-8 -*-
# create: @time: 1/5/23 10:51
import torch
import paddle
import numpy as np
from PIL import Image
from paddlenlp.transformers.ernie_layout.modeling import ErnieLayoutModel as PaddleErnieLayoutModel
from networks.model_util import ernie_cls_processing
from networks import ErnieLayoutConfig, ErnieLayoutModel, \
    ErnieLayoutImageProcessor, ErnieLayoutTokenizerFast, ERNIELayoutProcessor

paddle_model_name_or_path = "ernie-layoutx-base-uncased"
torch_model_or_path = "path/to/pretrained/model"
doc_imag_path = "./dummy_input.jpeg"


# initialize tokenizer
tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=torch_model_or_path)
context = ['This is an example document', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # all boxes are resized between 0 - 1000
pil_image = Image.open(doc_imag_path).convert("RGB")

# initialize feature extractor
feature_extractor = ErnieLayoutImageProcessor(apply_ocr=False)
processor = ERNIELayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)


context_encodings = processor(pil_image, "".join(context))
pixel_values_in_torch = torch.tensor(np.array(context_encodings.data['pixel_values']))

tokenized_res = ernie_cls_processing(context_encodings, layout)

# Prepare torch input
torch_input = dict(input_ids=torch.tensor([context_encodings.data["input_ids"]]),
                   bbox=torch.tensor([tokenized_res['bbox']]),
                   image=pixel_values_in_torch)

# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=torch_model_or_path)

torch_model = ErnieLayoutModel.from_pretrained(
    pretrained_model_name_or_path=torch_model_or_path,
    config=config,
)
torch_model.eval()

# Prepare paddlepaddle input
paddle_input = dict(input_ids=paddle.to_tensor([context_encodings.data["input_ids"]]),
                    bbox=paddle.to_tensor([tokenized_res['bbox']]),
                    image=paddle.to_tensor(pixel_values_in_torch.detach().cpu().numpy()))

paddle_model = PaddleErnieLayoutModel.from_pretrained(paddle_model_name_or_path)
paddle_model.eval()

# EVALUATE !!!
torch_output = torch_model(**torch_input)

paddle_output = paddle_model(**paddle_input)

# compare output between paddle and torch
torch_sequence_output, torch_pooled_output = torch_output
paddle_sequence_output, paddle_pooled_output = paddle_output

eps_pooled_output = np.square(np.subtract(torch_pooled_output.detach().cpu().numpy(), paddle_pooled_output.numpy())).mean()
eps_sequence_output = np.square(np.subtract(torch_sequence_output.detach().cpu().numpy(), paddle_sequence_output.numpy())).mean()

print("eps pooled output: {}; eps sequence output: {}".format(eps_pooled_output, eps_sequence_output))