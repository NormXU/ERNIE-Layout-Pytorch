# ERNIE-Layout-Pytorch

This is an unofficial Pytorch implementation of [ERNIE-Layout](http://arxiv.org/abs/2210.06155) which is originally released through [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP).


A Pytorch-style ERNIE-Layout Pretrained Model can be downloaded at [hub](https://huggingface.co/Norm/ERNIE-Layout-Pytorch/tree/main). The model is translated from [PaddlePaddle/ernie-layoutx-base-uncased](https://huggingface.co/PaddlePaddle/ernie-layoutx-base-uncased) with ``tools/convert2torch.py``. It is a script to convert all state dicts of pretrained models for PaddlePaddle into Pytorch style. Feel free to edit it if necessary.


### NEWs
**July 11, 2023** - Rewrite the processor for end-to-end preprocessing on bbox, question and label 
**Feb 16, 2023** - Make the tokenizer more huggingface-like with XLNetTokenizer following the advice from [maxjeblick](https://github.com/NormXU/ERNIE-Layout-Pytorch/issues/5). If you pull the latest codes and then find an error when loading the pretrained models, please replace ``"model_type": "xlnet"`` in corresponding ``config.json``. Also, you need to remove ``max_position_embeddings`` in ``config.json``. Or you can simply pull the latest configuration file from huggingface

### A Quick Example
```python
import torch
from PIL import Image
import torch.nn.functional as F
from networks import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering, \
    ErnieLayoutProcessor, ErnieLayoutTokenizerFast
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
doc_imag_path = "./dummy_input.jpeg"

context = ['This is an example sequence', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure  all boxes are normalized between 0 - 1000
pil_image = Image.open(doc_imag_path).convert("RGB")

# initialize tokenizer
tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)

# initialize feature extractor
feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
processor = ErnieLayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

# Tokenize context & questions
question = "what is it?"
encoding = processor(pil_image, question, context, boxes=layout, return_tensors="pt")

# dummy answer start && end index
start_positions = torch.tensor([6])
end_positions = torch.tensor([12])

# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
config.num_classes = 2  # start and end

# initialize ERNIE for VQA
model = ErnieLayoutForQuestionAnswering.from_pretrained(
    pretrained_model_name_or_path=pretrain_torch_model_or_path,
    config=config,
)

output = model(**encoding, start_positions=start_positions, end_positions=end_positions)

# decode output
start_max = torch.argmax(F.softmax(output.start_logits, dim=-1))
end_max = torch.argmax(F.softmax(output.end_logits, dim=-1)) + 1  # add one ##because of python list indexing
answer = tokenizer.decode(encoding.input_ids[0][start_max: end_max])
print(answer)

```
more examples can be found in ``examples`` folder

### Compare with Paddle Version
``examples/compare_output.py`` is a script to evaluate the MSE between paddle version output and the torch version output with the same dummpy input.

eps of pooled output: **0.00417756475508213**; eps of sequence output: **3.1264463674213205e-12**