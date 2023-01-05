# -*- coding:utf-8 -*-
# create: @time: 1/5/23 10:51
import torch
import paddle
import numpy as np
from PIL import Image
from paddlenlp.transformers.ernie_layout.modeling import ErnieLayoutModel as PaddleErnieLayoutModel
from networks.model_util import ernie_tokenize_layout
from networks.feature_extractor import ErnieFeatureExtractor
from networks.tokenizer import ErnieLayoutTokenizer
from networks.modeling_erine_layout import ErnieLayoutConfig, ErnieLayoutModel

paddle_model_name_or_path = "ernie-layoutx-base-uncased"
torch_model_or_path = "path/to/pretrained/model"
doc_imag_path = "./dummy_input.jpeg"

# initialize tokenizer
tokenizer = ErnieLayoutTokenizer.from_pretrained(pretrained_model_name_or_path=torch_model_or_path)
context = ['This is an example document', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # all boxes are resized between 0 - 1000

# initialize feature extractor
feature_extractor = ErnieFeatureExtractor()

tokenized_res = ernie_tokenize_layout(tokenizer, context, layout, labels=None)
pixel_values_in_torch = feature_extractor(Image.open(doc_imag_path).convert("RGB")).unsqueeze(0)
torch_input = dict(input_ids=torch.tensor([tokenized_res['input_ids']]),
                   bbox=torch.tensor([tokenized_res['bbox']]),
                   image=pixel_values_in_torch)

paddle_input = dict(input_ids=paddle.to_tensor([tokenized_res['input_ids']]),
                    bbox=paddle.to_tensor([tokenized_res['bbox']]),
                    image=paddle.to_tensor(pixel_values_in_torch.detach().cpu().numpy()))

paddle_model = PaddleErnieLayoutModel.from_pretrained(paddle_model_name_or_path)
paddle_model.eval()
paddle_output = paddle_model(**paddle_input)

# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=torch_model_or_path)

torch_model = ErnieLayoutModel.from_pretrained(
    pretrained_model_name_or_path=torch_model_or_path,
    config=config,
)
torch_model.eval()
torch_output = torch_model(**torch_input)

# compare output between paddle and torch
torch_sequence_output, torch_pooled_output = torch_output
paddle_sequence_output, paddle_pooled_output = paddle_output

eps_pooled_output = np.square(np.subtract(torch_pooled_output.detach().cpu().numpy(), paddle_pooled_output.numpy())).mean()
eps_sequence_output = np.square(np.subtract(torch_sequence_output.detach().cpu().numpy(), paddle_sequence_output.numpy())).mean()

print("eps pooled output: {}; eps sequence output: {}".format(eps_pooled_output, eps_sequence_output))