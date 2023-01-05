# -*- coding:utf-8 -*-
# encoding=utf-8
import torch
import paddle
import json
import os.path
from collections import OrderedDict
from paddlenlp.transformers import AutoModelForQuestionAnswering


model_name_or_path = "ernie-layoutx-base-uncased"

# set the path you want to save the ernie layout model
pretrain_torch_model_or_path = "/path/to/ernie-layoutx/torch_version"


def convert_weights():
    num_labels = 2
    # Load Model and Tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path,
                                                          num_classes=num_labels)
    state_dict = OrderedDict()
    base_model_prefix = model.base_model_prefix
    redundant_state_dict = dict()
    for sublayer in model.ernie_layout.named_sublayers():
        if hasattr(sublayer[1], 'weight') and sublayer[1].weight is not None:
            name = f'{base_model_prefix}.{sublayer[0]}.weight'
            params = torch.tensor(sublayer[1].weight.cpu().detach().numpy())
            if isinstance(sublayer[1], paddle.nn.Linear):
                params = torch.transpose(params, 0, 1)
            state_dict[name] = params
        if hasattr(sublayer[1], 'bias') and sublayer[1].bias is not None:
            name = f'{base_model_prefix}.{sublayer[0]}.bias'
            state_dict[name] = torch.tensor(sublayer[1].bias.cpu().detach().numpy())
        redundant_state_dict[sublayer[0]] = sublayer[1]

    # append some missing params
    state_dict[f'{base_model_prefix}.embeddings.position_ids'] = \
        torch.tensor(model.ernie_layout.embeddings.position_ids.cpu().detach().numpy())
    state_dict[f'{base_model_prefix}.encoder.rel_pos_x_bias.weight'] = \
        torch.transpose(
            torch.tensor(model.ernie_layout.encoder.rel_pos_x_bias.cpu().detach().numpy()), 0, 1)
    state_dict[f'{base_model_prefix}.encoder.rel_pos_y_bias.weight'] = \
        torch.transpose(
            torch.tensor(model.ernie_layout.encoder.rel_pos_y_bias.cpu().detach().numpy()), 0, 1)
    state_dict[f'{base_model_prefix}.encoder.rel_pos_bias.weight'] = \
        torch.transpose(
            torch.tensor(model.ernie_layout.encoder.rel_pos_bias.cpu().detach().numpy()), 0, 1)
    state_dict[f'{base_model_prefix}.visual.pixel_std'] = \
        torch.tensor(model.ernie_layout.visual.pixel_std.cpu().detach().numpy())
    state_dict[f'{base_model_prefix}.visual.pixel_mean'] = \
        torch.tensor(model.ernie_layout.visual.pixel_mean.cpu().detach().numpy())
    state_dict[f'{base_model_prefix}.visual.backbone.batch_norm1.running_var'] = \
        torch.tensor(model.ernie_layout.visual.backbone.conv._batch_norm._variance.cpu().detach().numpy())
    state_dict[f'{base_model_prefix}.visual.backbone.batch_norm1.running_mean'] = \
        torch.tensor(model.ernie_layout.visual.backbone.conv._batch_norm._mean.cpu().detach().numpy())

    state_dict[f'{base_model_prefix}.visual.backbone.resnet.layer0.0.shortcut.1.running_var'] = \
        torch.tensor(model.ernie_layout.visual.backbone.res2a.short._batch_norm._variance.cpu().detach().numpy())
    state_dict[f'{base_model_prefix}.visual.backbone.resnet.layer0.0.shortcut.1.running_mean'] = \
        torch.tensor(model.ernie_layout.visual.backbone.res2a.short._batch_norm._mean.cpu().detach().numpy())
    for idx, prefix in enumerate(['res2a', 'res2b', 'res2c']):
        for i in range(3):
            state_dict[f'{base_model_prefix}.visual.backbone.resnet.layer0.{idx}.batch_norm{i + 1}.running_var'] = \
                torch.tensor(
                    getattr(getattr(model.ernie_layout.visual.backbone,
                                    prefix),
                            f'conv{i}')._batch_norm._variance.cpu().detach().numpy())
            state_dict[f'{base_model_prefix}.visual.backbone.resnet.layer0.{idx}.batch_norm{i + 1}.running_mean'] = \
                torch.tensor(
                    getattr(getattr(model.ernie_layout.visual.backbone,
                                    prefix),
                            f'conv{i}')._batch_norm._mean.cpu().detach().numpy())

    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if 'visual.backbone' in name:
            # conv1
            name = name.replace('conv._conv.weight', 'conv1.weight')
            # batchnorm1
            name = name.replace('conv._batch_norm.weight', 'batch_norm1.weight')
            name = name.replace('conv._batch_norm.bias', 'batch_norm1.bias')
            if 'res2a' in name:
                name = name.replace('res2a.conv0._conv.weight', 'resnet.layer0.0.conv1.weight')
                name = name.replace('res2a.conv0._batch_norm.weight', 'resnet.layer0.0.batch_norm1.weight')
                name = name.replace('res2a.conv0._batch_norm.bias', 'resnet.layer0.0.batch_norm1.bias')

                name = name.replace('res2a.conv1._conv.weight', 'resnet.layer0.0.conv2.weight')
                name = name.replace('res2a.conv1._batch_norm.weight', 'resnet.layer0.0.batch_norm2.weight')
                name = name.replace('res2a.conv1._batch_norm.bias', 'resnet.layer0.0.batch_norm2.bias')

                name = name.replace('res2a.conv2._conv.weight', 'resnet.layer0.0.conv3.weight')
                name = name.replace('res2a.conv2._batch_norm.weight', 'resnet.layer0.0.batch_norm3.weight')
                name = name.replace('res2a.conv2._batch_norm.bias', 'resnet.layer0.0.batch_norm3.bias')

                # shortcut weights: CONV + BN
                name = name.replace('res2a.short._conv.weight', 'resnet.layer0.0.shortcut.0.weight')
                name = name.replace('res2a.short._batch_norm.weight', 'resnet.layer0.0.shortcut.1.weight')
                name = name.replace('res2a.short._batch_norm.bias', 'resnet.layer0.0.shortcut.1.bias')

            if 'res2b' in name:
                name = name.replace('res2b.conv0._conv.weight', 'resnet.layer0.1.conv1.weight')
                name = name.replace('res2b.conv0._batch_norm.weight', 'resnet.layer0.1.batch_norm1.weight')
                name = name.replace('res2b.conv0._batch_norm.bias', 'resnet.layer0.1.batch_norm1.bias')

                name = name.replace('res2b.conv1._conv.weight', 'resnet.layer0.1.conv2.weight')
                name = name.replace('res2b.conv1._batch_norm.weight', 'resnet.layer0.1.batch_norm2.weight')
                name = name.replace('res2b.conv1._batch_norm.bias', 'resnet.layer0.1.batch_norm2.bias')

                name = name.replace('res2b.conv2._conv.weight', 'resnet.layer0.1.conv3.weight')
                name = name.replace('res2b.conv2._batch_norm.weight', 'resnet.layer0.1.batch_norm3.weight')
                name = name.replace('res2b.conv2._batch_norm.bias', 'resnet.layer0.1.batch_norm3.bias')

            if 'res2c' in name:
                name = name.replace('res2c.conv0._conv.weight', 'resnet.layer0.2.conv1.weight')
                name = name.replace('res2c.conv0._batch_norm.weight', 'resnet.layer0.2.batch_norm1.weight')
                name = name.replace('res2c.conv0._batch_norm.bias', 'resnet.layer0.2.batch_norm1.bias')

                name = name.replace('res2c.conv1._conv.weight', 'resnet.layer0.2.conv2.weight')
                name = name.replace('res2c.conv1._batch_norm.weight', 'resnet.layer0.2.batch_norm2.weight')
                name = name.replace('res2c.conv1._batch_norm.bias', 'resnet.layer0.2.batch_norm2.bias')

                name = name.replace('res2c.conv2._conv.weight', 'resnet.layer0.2.conv3.weight')
                name = name.replace('res2c.conv2._batch_norm.weight', 'resnet.layer0.2.batch_norm3.weight')
                name = name.replace('res2c.conv2._batch_norm.bias', 'resnet.layer0.2.batch_norm3.bias')
        new_state_dict[name] = param
    if not os.path.exists(pretrain_torch_model_or_path):
        os.mkdir(pretrain_torch_model_or_path)
    torch.save(new_state_dict, os.path.join(pretrain_torch_model_or_path, "pytorch_model.bin"))
    with open(os.path.join(pretrain_torch_model_or_path, "config.json"), "w") as outfile:
        config = model.ernie_layout.init_config
        config['init_args'] = (config['init_args'][0].__dict__)
        outfile.write(json.dumps(config))


if __name__ == '__main__':
    convert_weights()
