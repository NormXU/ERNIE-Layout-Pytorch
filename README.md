# ERNIE-Layout-Pytorch

This is an unofficial Pytorch implementation of [ERNIE-Layout](http://arxiv.org/abs/2210.06155) which is originally released through PaddleNLP.

``tools/convert2torch.py`` is a converting script to convert all state dicts of pretrained models for PaddlePaddle into Pytorch style. Feel free to edit it if necessary.


A Pytorch-style ERNIE-Layout Pretrained Model can be downloaded at [hub](https://huggingface.co/Norm/ERNIE-Layout-Pytorch/tree/main)


**A Quick Example**
```python
from networks.modeling_erine_layout import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering
from networks.feature_extractor import ErnieFeatureExtractor
from networks.tokenizer import ErnieLayoutTokenizer
from networks.model_util import ernie_qa_tokenize, prepare_context_info
from PIL import Image


pretrain_torch_model_or_path = "path/to/pretrained-model"

# initialize tokenizer
tokenizer = ErnieLayoutTokenizer.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
context = ['This is an example document', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]

# intialize feature extractor
feature_extractor = ErnieFeatureExtractor()

# Tokenize context & questions
context_encodings, = prepare_context_info(tokenizer, context, layout)
question = "what is it?"
tokenized_res = ernie_qa_tokenize(tokenizer, question, context_encodings)

# answer start && end index
tokenized_res['start_positions'] = 6
tokenized_res['end_positions'] = 12

# open the image of the document
pil_image = Image.open("/path/to/image").convert("RGB")

# Process image
tokenized_res['pixel_values'] = feature_extractor(pil_image)


# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
config.num_classes = 2 # start and end

# initialize ERNIE for VQA
model = ErnieLayoutForQuestionAnswering.from_pretrained(
    pretrained_model_name_or_path=pretrain_torch_model_or_path,
    config=config,
)

output = model(**tokenized_res)


```
