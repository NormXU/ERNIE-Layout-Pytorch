# -*- coding:utf-8 -*-
# create: @time: 2/16/23 16:46
import os
from transformers import XLNetTokenizerFast


class ErnieLayoutTokenizerFast(XLNetTokenizerFast):
    def __init__(self,
                 vocab_file=None,
                 tokenizer_file=None,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=False,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 additional_special_tokens=["<eop>", "<eod>"],
                 **kwargs
                 ):
        vocab_file = os.path.join(kwargs["name_or_path"], vocab_file)
        super().__init__(vocab_file=vocab_file,
                         tokenizer_file=tokenizer_file,
                         do_lower_case=do_lower_case,
                         remove_space=remove_space,
                         keep_accents=keep_accents,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         additional_special_tokens=additional_special_tokens,
                         **kwargs)
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = self.vocab_size - 1
