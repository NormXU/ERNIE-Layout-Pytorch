# ERNIE-Layout-Pytorch

This is an unofficial Pytorch implementation of [ERNIE-Layout](http://arxiv.org/abs/2210.06155) which is originally released through [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP).


A Pytorch-style ERNIE-Layout Pretrained Model can be downloaded at [hub](https://huggingface.co/Norm/ERNIE-Layout-Pytorch/tree/main). The model is translated from ``Taskflow("document_intelligence", lang="cn")`` of PaddlePaddle.

If you are looking for ERNIE-layout in English, please use ``tools/convert2torch.py``. It is a converting script to convert all state dicts of pretrained models for PaddlePaddle into Pytorch style. Feel free to edit it if necessary.

### A Quick Example
```python
import torch
from networks.modeling_erine_layout import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering
from networks.feature_extractor import ErnieFeatureExtractor
from networks.tokenizer import ErnieLayoutTokenizer
from networks.model_util import ernie_qa_tokenize, prepare_context_info
from PIL import Image


pretrain_torch_model_or_path = "path/to/pretrained/mode"
doc_imag_path = "path/to/doc/image"

device = torch.device("cuda:0")

# initialize tokenizer
tokenizer = ErnieLayoutTokenizer.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
context = ['This is an example document', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # all boxes are resized between 0 - 1000

# initialize feature extractor
feature_extractor = ErnieFeatureExtractor()

# Tokenize context & questions
context_encodings = prepare_context_info(tokenizer, context, layout)
question = "what is it?"
tokenized_res = ernie_qa_tokenize(tokenizer, question, context_encodings)
tokenized_res['input_ids'] = torch.tensor([tokenized_res['input_ids']]).to(device)
tokenized_res['bbox'] = torch.tensor([tokenized_res['bbox']]).to(device)

# answer start && end index
tokenized_res['start_positions'] = torch.tensor([6]).to(device)
tokenized_res['end_positions'] = torch.tensor([12]).to(device)


# open the image of the document and process image
tokenized_res['pixel_values'] = feature_extractor(Image.open(doc_imag_path).convert("RGB")).unsqueeze(0).to(device)


# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
config.num_classes = 2 # start and end

# initialize ERNIE for VQA
model = ErnieLayoutForQuestionAnswering.from_pretrained(
    pretrained_model_name_or_path=pretrain_torch_model_or_path,
    config=config,
)
model.to(device)

output = model(**tokenized_res)

```
more examples can be found in ``examples`` folder

### Compare with Paddle Version
``examples/compare_output.py`` is a script to evaluate the MSE between paddle version output and the torch version output with the same dummpy input.

eps of pooled output: **0.004253871738910675**; eps of sequence output: **3.5654803762219522e-12**