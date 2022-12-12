# ERNIE-Layout-Pytorch

This is an unofficial Pytorch implementation of [ERNIE-Layout](http://arxiv.org/abs/2210.06155) which is originally released through PaddleNLP.

``tools/convert2torch.py`` is a converting script to convert all state dicts of pretrained models for PaddlePaddle into Pytorch style. Feel free to edit it if necessary.


A Pytorch-style ERNIE-Layout Pretrained Model can be downloaded at [hub](https://huggingface.co/Norm/ERNIE-Layout-Pytorch/tree/main)

**Get Ready**

After downloading the model, you need to set ``sentencepiece_model_file`` and ``vocab_file`` in **tokenizer_config.json** to your own directory

**A Quick Example**
```python
from networks.modeling_erine_layout import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering
from networks.tokenizer import ErnieLayoutTokenizer


pretrain_torch_model_or_path = "path/to/pretrained-model"

# initialize tokenizer
tokenizer = ErnieLayoutTokenizer.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
encodings = tokenizer.encode_plus(text="Question", text_pair="Answer", add_special_tokens=True,
                                  max_length=512, truncation="only_second",
                                  return_offsets_mapping=True, return_attention_mask=True,
                                  return_overflowing_tokens=True)

# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
config.num_classes = 2

# initialize ERNIE for VQA
model = ErnieLayoutForQuestionAnswering.from_pretrained(
    pretrained_model_name_or_path=pretrain_torch_model_or_path,
    config=config,
)

```
