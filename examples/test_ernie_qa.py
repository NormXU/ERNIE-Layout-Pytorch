import torch
from networks.modeling_erine_layout import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering
from networks.feature_extractor import ErnieFeatureExtractor
from networks.tokenizer import ErnieLayoutTokenizer
from networks.model_util import ernie_qa_tokenize, prepare_context_info
from PIL import Image


pretrain_torch_model_or_path = "path/to/pretrained/model"
doc_imag_path = "path/to/doc/image"

device = torch.device("cuda:0")

def main():
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

if __name__ == '__main__':
    main()