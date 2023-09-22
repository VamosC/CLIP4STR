#!/bin/bash
### training on real dataset
python ../train.py +experiment=vl4str-large model=vl4str dataset=real \
                    model.lr=8.4e-5 model.batch_size=48  \
                    trainer.accumulate_grad_batches=5 trainer.max_epochs=10 trainer.gpus=4 \
                    trainer.val_check_interval=10000

### training on synthetic dataset
# python ../train.py +experiment=vl4str-large model=vl4str dataset=synth \
#                     model.lr=8.4e-5 model.batch_size=48  \
#                     trainer.accumulate_grad_batches=5 trainer.max_epochs=6 trainer.gpus=4 \
#                     trainer.val_check_interval=10000


# loading checkpoint from clip/ViT-L-14.pt
#    | Name                       | Type              | Params
# ------------------------------------------------------------------
# 0  | clip_model                 | CLIP              | 427 M 
# 1  | clip_model.visual          | VisionTransformer | 303 M 
# 2  | clip_model.transformer     | Transformer       | 85.1 M
# 3  | clip_model.token_embedding | Embedding         | 37.9 M
# 4  | clip_model.ln_final        | LayerNorm         | 1.5 K 
# 5  | visual_decoder             | Decoder           | 9.6 M 
# 6  | visual_decoder.layers      | ModuleList        | 9.5 M 
# 7  | visual_decoder.text_embed  | TokenEmbedding    | 74.5 K
# 8  | visual_decoder.norm        | LayerNorm         | 1.5 K 
# 9  | visual_decoder.dropout     | Dropout           | 0     
# 10 | visual_decoder.head        | Linear            | 73.1 K
# 11 | cross_decoder              | Decoder           | 9.6 M 
# 12 | cross_decoder.layers       | ModuleList        | 9.5 M 
# 13 | cross_decoder.text_embed   | TokenEmbedding    | 74.5 K
# 14 | cross_decoder.norm         | LayerNorm         | 1.5 K 
# 15 | cross_decoder.dropout      | Dropout           | 0     
# 16 | cross_decoder.head         | Linear            | 73.1 K
# ------------------------------------------------------------------
# 366 M     Trainable params
# 80.5 M    Non-trainable params
# 446 M     Total params
# 1,787.445 Total estimated model params size (MB)




