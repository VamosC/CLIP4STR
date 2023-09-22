#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
ckpt_id=$2

### DEFINE THE ROOT PATH HERE ###
abs_root=/PUT/YOUR/PATH/HERE

# exp_path=${abs_root}/output/${folder_id}/checkpoints/last.ckpt
exp_path=${abs_root}/output/${ckpt_id}
# exp_path=${abs_root}/output/parseq-bb5792a6.pt
runfile=${abs_root}/code/clip4str/test.py
# clip_model_path=${abs_root}/pretrained/clip/ViT-B-16.pt
clip_model_path=${abs_root}/pretrained/clip/ViT-L-14.pt
data_root=${abs_root}/dataset/str_dataset


python ${runfile} ${exp_path} \
            --data_root ${data_root} \
            --clip_model_path ${clip_model_path} \
            --new
            # --clip_refine \
            # --sample_K 5 \
            # --sample_K2 3 \
            # --sample_total 50 \
            # --alpha 0.1 \
