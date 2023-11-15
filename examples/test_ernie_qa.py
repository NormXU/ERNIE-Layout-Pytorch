import torch
import torch.nn.functional as F
from PIL import Image
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

from networks import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering, \
    ErnieLayoutProcessor, ErnieLayoutTokenizerFast

pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
doc_imag_path = "./dummy_input.jpeg"


def main():
    context = ['This is an example sequence', 'All ocr boxes are inserted into this list']
    layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure  all boxes are normalized between 0 - 1000
    pil_image = Image.open(doc_imag_path).convert("RGB")

    # initialize tokenizer
    tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
    # setting tokenizer to pad on the right side of the sequence
    tokenizer.padding_side = 'right'

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


if __name__ == '__main__':
    main()
