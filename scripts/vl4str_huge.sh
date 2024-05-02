#!/bin/bash

### training on real dataset
# python ../train.py +experiment=vl4str-huge model=vl4str dataset=real \
#                     model.lr=8.4e-5 model.batch_size=16  \
#                     trainer.accumulate_grad_batches=8 trainer.max_epochs=8 trainer.gpus=8 \
#                     trainer.val_check_interval=10000 \
#                     model.clip_pretrained=/home/shuai/pretrained/clip/appleDFN5B-CLIP-ViT-H-14.bin


### Commands of Shuai to train a CLIP4STR-H model with OpenCLIP weights on RBU(6.5M)
# export CUDA_VISIBLE_DEVICES=4,5,6,7
python ../train.py +experiment=vl4str-huge model=vl4str dataset=real \
                    data.root_dir=/home/shuai/dataset/str_dataset_ub \
                    model.lr=8.4e-5 model.batch_size=32  \
                    trainer.accumulate_grad_batches=8 trainer.max_epochs=4 trainer.gpus=4 \
                    trainer.val_check_interval=10000 \
                    model.clip_pretrained=/home/shuai/pretrained/clip/appleDFN5B-CLIP-ViT-H-14.bin
                    # model.clip_pretrained=/home/shuai/pretrained/clip/OpenCLIP-ViT-H-14-laion2B-s32B-b79K.bin

# The dimension of the visual decoder is 1024.
#    | Name                       | Type              | Params
# ------------------------------------------------------------------
# 0  | clip_model                 | CLIP              | 986 M
# 1  | clip_model.visual          | VisionTransformer | 632 M
# 2  | clip_model.transformer     | Transformer       | 302 M
# 3  | clip_model.token_embedding | Embedding         | 50.6 M
# 4  | clip_model.ln_final        | LayerNorm         | 2.0 K
# 5  | visual_decoder             | Decoder           | 17.0 M
# 6  | visual_decoder.layers      | ModuleList        | 16.8 M
# 7  | visual_decoder.text_embed  | TokenEmbedding    | 99.3 K
# 8  | visual_decoder.norm        | LayerNorm         | 2.0 K
# 9  | visual_decoder.dropout     | Dropout           | 0
# 10 | visual_decoder.head        | Linear            | 97.4 K
# 11 | cross_decoder              | Decoder           | 17.0 M
# 12 | cross_decoder.layers       | ModuleList        | 16.8 M
# 13 | cross_decoder.text_embed   | TokenEmbedding    | 99.3 K
# 14 | cross_decoder.norm         | LayerNorm         | 2.0 K
# 15 | cross_decoder.dropout      | Dropout           | 0
# 16 | cross_decoder.head         | Linear            | 97.4 K
# ------------------------------------------------------------------
# 893 M     Trainable params
# 126 M     Non-trainable params
# 1.0 B     Total params

