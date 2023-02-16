# -*- coding:utf-8 -*-
# create: @time: 12/28/22 16:44

def ernie_qa_processing(tokenizer, question, bbox,
                        context_encodings,
                        cls_token_box=[0, 0, 0, 0],
                        sep_token_box=[1000, 1000, 1000, 1000]):
    encoding = dict(
        input_ids=tokenizer.encode(question, add_special_tokens=True),
        token_list=tokenizer.tokenize(question))
    encoding['sequence_ids'] = [None] + [0] * len(encoding['token_list']) + [None]
    encoding['word_ids'] = [None] + [0] * len(encoding['token_list']) + [None]
    encoding['bbox'] = [[0, 0, 0, 0]] * (len(encoding['input_ids']))
    encoding['sequence_ids'].extend([None])
    encoding['word_ids'].extend([None])
    encoding['input_ids'].extend([tokenizer.cls_token_id])
    encoding['bbox'].append(cls_token_box)
    word_id = 0
    for (idx, ctx) in enumerate(context_encodings.encodings):
        seq_len = len(ctx.ids[1:-1])
        encoding['sequence_ids'].extend([1] * seq_len)
        encoding['word_ids'].extend([word_id] * seq_len)
        word_id += 1
        encoding['input_ids'].extend(ctx.ids[1:-1])
        encoding['token_list'].extend(ctx.type_ids[1:-1])
        encoding['bbox'].extend([bbox[idx]] * seq_len)
    encoding['input_ids'].extend([tokenizer.sep_token_id])
    encoding['sequence_ids'].extend([None])
    encoding['word_ids'].extend([None])
    encoding['bbox'].append(sep_token_box)
    encoding['context_encodings'] = context_encodings
    return encoding


def ernie_cls_processing(context_encodings, layout, labels=None, cls_token_id=0, sep_token_id=2):
    label_pad_token_id = -100
    cls_token_box = [0, 0, 0, 0]
    sep_token_box = [1000, 1000, 1000, 1000]

    tokenized_res = dict(input_ids=[cls_token_id],
                         token_list=list(),
                         labels=[0] if labels is not None else None,
                         bbox=[cls_token_box])
    for idx, encoding in enumerate(context_encodings.encodings):
        seq_len = len(encoding.ids[1:-1])
        tokenized_res['input_ids'].extend(encoding.ids[1:-1])
        tokenized_res['token_list'].extend(encoding.type_ids[1:-1])
        tokenized_res['bbox'].extend([layout[idx]] * seq_len)
        if labels:
            if labels[idx] != 0:
                tokenized_res['labels'].extend([labels[idx]] + [label_pad_token_id] * (seq_len - 1))
            else:
                tokenized_res['labels'].extend([labels[idx]] * seq_len)
    tokenized_res['bbox'].append(sep_token_box)
    tokenized_res['input_ids'].append(sep_token_id)
    if labels:
        tokenized_res['labels'].append(0)
    return tokenized_res
