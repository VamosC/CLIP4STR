#!/bin/bash

### re-implementation of PARSEQ with ViT-S
# python ../train.py +experiment=parseq model=parseq 

### re-implementation of PARSEQ with ViT-B
python ../train.py +experiment=parseq-base model=parseq \
                    trainer.max_epochs=16 trainer.gpus=4 model.batch_size=256 \
                    model.lr=8.4e-5

# [ViT] Load pretrained weights from ImageNet-21K pretrained model (vit_base_patch16_224_miil_21k.pth)
# missing_keys:  ['blocks.0.attn.qkv.bias', 'blocks.1.attn.qkv.bias', 'blocks.2.attn.qkv.bias', 'blocks.3.attn.qkv.bias', 'blocks.4.attn.qkv.bias', 'blocks.5.attn.qkv.bias', 'blocks.6.attn.qkv.bias', 'blocks.7.attn.qkv.bias', 'blocks.8.attn.qkv.bias', 'blocks.9.attn.qkv.bias', 'blocks.10.attn.qkv.bias', 'blocks.11.attn.qkv.bias']
# unexpected_keys:  []
#   | Name       | Type           | Params
# ----------------------------------------------
# 0 | encoder    | Encoder        | 85.8 M
# 1 | proj       | Linear         | 393 K 
# 2 | decoder    | Decoder        | 4.2 M 
# 3 | head       | Linear         | 48.7 K
# 4 | text_embed | TokenEmbedding | 49.7 K
# 5 | dropout    | Dropout        | 0     
# ----------------------------------------------
# 90.5 M    Trainable params
# 0         Non-trainable params
# 90.5 M    Total params
# 362.035   Total estimated model params size (MB)
