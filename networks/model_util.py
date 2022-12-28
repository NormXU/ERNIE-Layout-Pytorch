# -*- coding:utf-8 -*-
# create: @time: 12/28/22 16:44

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    try:
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248

            rstring += chr(inside_code)
    except Exception:
        pass
    return rstring


def count_blanks(input_str, start_idx):
    blank_char = ' '
    num_blanks = 0
    while start_idx < len(input_str) and input_str[start_idx] == blank_char:
        num_blanks += 1
        start_idx += 1
    return num_blanks


def prepare_context_info(tokenizer, context, layout, debug=True):
    """
        Project context index to subword index; Version 2
    """
    blank_char = ' '
    underline_char = '▁'  # xlmRobertaTokenzier use this char to represent the beginning of a sentence
    context_encodings = []
    for text_chunk, bbox in zip(context, layout):
        encoding = dict(  # offset_mapping=tokenizer.get_offset_mapping(text_chunk),
            input_ids=tokenizer.encode(text_chunk, add_special_tokens=False),
            token_list=tokenizer.tokenize(text_chunk))
        encoding.update({'bbox': [bbox] * len(encoding['input_ids'])})
        context_encodings.append(encoding)

    context_id2subword_id = []
    debug_context_id2subword_id = []
    char_len_cnt = 0
    all_missing_tail_blank = 0
    for ctx_idx, (ctx, chunk_encoding) in enumerate(zip(context, context_encodings)):
        tmp_offset = []
        sop_idx = 0
        encoding_token_list = chunk_encoding['token_list']
        for t_id, token in enumerate(encoding_token_list):
            eop_idx = sop_idx + len(token)
            token = token.replace(underline_char, '')
            while strQ2B(ctx[sop_idx:eop_idx]) != strQ2B(token) and sop_idx < eop_idx:
                eop_idx -= 1
            if sop_idx == eop_idx:
                # for ['_', ‘A’, ...]
                if ctx[sop_idx] == blank_char:
                    #  _ is a blank or are blanks.
                    num_blanks = count_blanks(ctx, sop_idx)
                    eop_idx += (len(token) + num_blanks)
                    current_offset = (sop_idx, eop_idx)
                else:
                    # [ _, A, B, _, CD] and both _ is a nonsense prefix.
                    # it encodes as [(0, 0) (0, 1) (1, 2) (0, 0) (2, 4)]
                    current_offset = (0, 0)
            else:
                # for ['_abc', ...]
                if ctx[sop_idx] == blank_char:
                    #  _ is a blank or are blanks.
                    num_blanks = count_blanks(ctx, sop_idx)
                    eop_idx += (len(token) + num_blanks)
                    current_offset = (sop_idx, eop_idx)
                else:
                    eop_idx = sop_idx + len(token)
                    current_offset = (sop_idx, eop_idx)

            tmp_offset.append(current_offset)
            sop_idx = eop_idx

        missing_tail_blank = len(ctx) - eop_idx
        bias = 1 if ctx_idx == 0 else 0
        for i, offset in enumerate(tmp_offset):
            valid_token = encoding_token_list[i]
            end_context_idx = offset[1] + char_len_cnt - bias + all_missing_tail_blank
            if debug:
                debug_context_id2subword_id.append({valid_token: end_context_idx})
            context_id2subword_id.append(end_context_idx)
        all_missing_tail_blank += missing_tail_blank
        char_len_cnt = eop_idx + char_len_cnt - bias

    return context_encodings


def ernie_qa_tokenize(tokenizer, question, context_encodings,
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
    encoding['input_ids'].extend([0])
    encoding['bbox'].append(cls_token_box)
    word_id = 0
    for ctx in context_encodings:
        encoding['sequence_ids'].extend([1] * len(ctx['input_ids']))
        encoding['word_ids'].extend([word_id] * len(ctx['token_list']))
        word_id += 1
        encoding['input_ids'].extend(ctx['input_ids'])
        encoding['token_list'].extend(ctx['token_list'])
        # encoding['offset_mapping'].extend(ctx['offset_mapping'])
        encoding['bbox'].extend(ctx['bbox'])
    encoding['input_ids'].extend([2])
    encoding['sequence_ids'].extend([None])
    encoding['word_ids'].extend([None])
    encoding['bbox'].append(sep_token_box)
    encoding['context_encodings'] = context_encodings
    return encoding
