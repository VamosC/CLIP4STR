"""
Implementation of ViTSTR based on timm VisionTransformer.

TODO:
1) distilled deit backbone
2) base deit backbone

Copyright 2021 Rowel Atienza
"""
import os
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_info
from timm.models.vision_transformer import VisionTransformer


class ViTSTR(VisionTransformer):
    """
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    """

    def forward(self, x, seqlen: int = 25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def load_pretrained_weight(model, pretrained):
    if not os.path.exists(pretrained):
        rank_zero_info("[ViT] pretrained weight ({}) does not exist".format(pretrained))
    rank_zero_info("[ViT] Load pretrained weights from ImageNet-21K pretrained model ({})".format(pretrained))
    state_dict = torch.load(pretrained, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # diff shapes
    # pos_embed torch.Size([1, 129, 768])
    # pos_embed torch.Size([1, 197, 768])
    # patch_embed.proj.weight torch.Size([768, 3, 4, 8])
    # patch_embed.proj.weight torch.Size([768, 3, 16, 16])
    # head.weight torch.Size([25, 768])
    # head.weight torch.Size([21843, 768])
    # head.bias torch.Size([25])
    # head.bias torch.Size([21843])
    # convert to fp32
    new_state_dict = {}
    for keys, value in state_dict.items():
        new_state_dict[keys] = value.float()

    # position embedding
    old_pe = new_state_dict['pos_embed']
    new_pe_len = model.pos_embed.shape[1]
    new_state_dict['pos_embed'] = old_pe[:, :new_pe_len, ...]

    # first conv
    old_conv1_w = new_state_dict['patch_embed.proj.weight']
    new_conv1_w_shape = model.patch_embed.proj.weight.shape
    if old_conv1_w.shape != new_conv1_w_shape:
        rank_zero_info("[ViT] averageing conv1 weight (patch_embed.proj) of ViT")
        # average across [H, W]
        kernel_size = (old_conv1_w.shape[-2] // new_conv1_w_shape[-2], old_conv1_w.shape[-1] // new_conv1_w_shape[-1])
        new_state_dict['patch_embed.proj.weight'] = nn.AvgPool2d(kernel_size, stride=kernel_size)(old_conv1_w)

    # head
    old_proj_w = new_state_dict['head.weight']
    old_proj_bias = new_state_dict['head.bias']
    if not isinstance(model.head, nn.Identity):
        new_output_dim = model.head.weight.shape[0]
        if new_output_dim != old_proj_w.shape[1]:
            rank_zero_info("[ViT] downsmaple the output head of ViT")
            new_state_dict['head.weight'] = nn.AdaptiveAvgPool1d(new_output_dim)(old_proj_w.transpose(0, 1)).transpose(0,1)
            if hasattr(model.head, 'bias'):
                new_state_dict['head.bias'] = nn.AdaptiveAvgPool1d(new_output_dim)(old_proj_bias.unsqueeze(0)).squeeze()
    else:
        del new_state_dict['head.weight']
        del new_state_dict['head.bias']

    # cls token
    if model.cls_token is None:
        del new_state_dict['cls_token']

    # load
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

# model1 = VisionTransformer(img_size=224, patch_size=16, depth=12, mlp_ratio=4, qkv_bias=True,
#                            embed_dim=768, num_heads=12, num_classes=21843)
# model1.load_pretrained("B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz")
# sd1 = model1.state_dict()
# torch.save(sd1, "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.pth")
