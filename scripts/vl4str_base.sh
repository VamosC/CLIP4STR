#!/bin/bash

### training on real dataset
python ../train.py +experiment=vl4str model=vl4str dataset=real \
                    trainer.max_epochs=16 trainer.gpus=8 \
                    model.lr=8.4e-5 \
                    model.batch_size=128
#                     trainer.accumulate_grad_batches=2


### training on synthetic dataset
# python ../train.py +experiment=vl4str model=vl4str dataset=synth \
#                     trainer.gpus=4 trainer.max_epochs=7 \
#                     model.lr=8.4e-5 \
#                     model.batch_size=128 trainer.accumulate_grad_batches=2


### training on real dataset with CLIP-B-32
# python ../train.py +experiment=vl4str-b32 model=vl4str dataset=real \
#                     trainer.gpus=4 trainer.max_epochs=16 \
#                     model.lr=8.4e-5 model.batch_size=256


# loading checkpoint from clip/ViT-B-16.pt
#    | Name                       | Type              | Params
# ------------------------------------------------------------------
# 0  | clip_model                 | CLIP              | 149 M 
# 1  | clip_model.visual          | VisionTransformer | 86.2 M
# 2  | clip_model.transformer     | Transformer       | 37.8 M
# 3  | clip_model.token_embedding | Embedding         | 25.3 M
# 4  | clip_model.ln_final        | LayerNorm         | 1.0 K 
# 5  | visual_decoder             | Decoder           | 4.3 M 
# 6  | visual_decoder.layers      | ModuleList        | 4.2 M 
# 7  | visual_decoder.text_embed  | TokenEmbedding    | 49.7 K
# 8  | visual_decoder.norm        | LayerNorm         | 1.0 K 
# 9  | visual_decoder.dropout     | Dropout           | 0     
# 10 | visual_decoder.head        | Linear            | 48.7 K
# 11 | cross_decoder              | Decoder           | 4.3 M 
# 12 | cross_decoder.layers       | ModuleList        | 4.2 M 
# 13 | cross_decoder.text_embed   | TokenEmbedding    | 49.7 K
# 14 | cross_decoder.norm         | LayerNorm         | 1.0 K 
# 15 | cross_decoder.dropout      | Dropout           | 0     
# 16 | cross_decoder.head         | Linear            | 48.7 K
# ------------------------------------------------------------------
# 114 M     Trainable params
# 44.3 M    Non-trainable params
# 158 M     Total params
# 633.025   Total estimated model params size (MB)


# loading checkpoint from clip/ViT-B-32.pt
# The dimension of the visual decoder is 512.
#    | Name                       | Type              | Params
# ------------------------------------------------------------------
# 0  | clip_model                 | CLIP              | 151 M 
# 1  | clip_model.visual          | VisionTransformer | 87.8 M
# 2  | clip_model.transformer     | Transformer       | 37.8 M
# 3  | clip_model.token_embedding | Embedding         | 25.3 M
# 4  | clip_model.ln_final        | LayerNorm         | 1.0 K 
# 5  | visual_decoder             | Decoder           | 4.3 M 
# 6  | visual_decoder.layers      | ModuleList        | 4.2 M 
# 7  | visual_decoder.text_embed  | TokenEmbedding    | 49.7 K
# 8  | visual_decoder.norm        | LayerNorm         | 1.0 K 
# 9  | visual_decoder.dropout     | Dropout           | 0     
# 10 | visual_decoder.head        | Linear            | 48.7 K
# 11 | cross_decoder              | Decoder           | 4.3 M 
# 12 | cross_decoder.layers       | ModuleList        | 4.2 M 
# 13 | cross_decoder.text_embed   | TokenEmbedding    | 49.7 K
# 14 | cross_decoder.norm         | LayerNorm         | 1.0 K 
# 15 | cross_decoder.dropout      | Dropout           | 0     
# 16 | cross_decoder.head         | Linear            | 48.7 K
# ------------------------------------------------------------------
# 115 M     Trainable params
# 44.3 M    Non-trainable params
# 159 M     Total params
# 639.652   Total estimated model params size (MB)