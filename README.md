# CLIP4STR

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-iiit5k)](https://paperswithcode.com/sota/scene-text-recognition-on-iiit5k?p=clip4str-a-simple-baseline-for-scene-text-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-svt)](https://paperswithcode.com/sota/scene-text-recognition-on-svt?p=clip4str-a-simple-baseline-for-scene-text-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-icdar2013)](https://paperswithcode.com/sota/scene-text-recognition-on-icdar2013?p=clip4str-a-simple-baseline-for-scene-text-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-icdar2015)](https://paperswithcode.com/sota/scene-text-recognition-on-icdar2015?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-svtp)](https://paperswithcode.com/sota/scene-text-recognition-on-svtp?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-cute80)](https://paperswithcode.com/sota/scene-text-recognition-on-cute80?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-host)](https://paperswithcode.com/sota/scene-text-recognition-on-host?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-wost)](https://paperswithcode.com/sota/scene-text-recognition-on-wost?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-coco-text)](https://paperswithcode.com/sota/scene-text-recognition-on-coco-text?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-ic19-art)](https://paperswithcode.com/sota/scene-text-recognition-on-ic19-art?p=clip4str-a-simple-baseline-for-scene-text-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip4str-a-simple-baseline-for-scene-text-1/scene-text-recognition-on-uber-text)](https://paperswithcode.com/sota/scene-text-recognition-on-uber-text?p=clip4str-a-simple-baseline-for-scene-text-1)



This is a dedicated re-implementation of [CLIP4STR: A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model
](https://arxiv.org/abs/2305.14014).


##  Table of Contents

<!--ts-->
* [Introduction](#Introduction)
* [Installation](#Installation)
* [Results](#Results)
* [Training](#Training) 
* [Inference](#Inference)
* [FAQs](#FAQs)
* [Citations](#Citations)
* [Acknowledgements](#Acknowledgements)
<!--te-->

## News

- [26/12/2024] CLIP4STR will appear at TIP! The early access version is available at https://ieeexplore.ieee.org/document/10816351. A final version is also at https://arxiv.org/abs/2305.14014.
- [02/05/2024] Add new CLIP4STR models pre-trained on DataComp-1B, LAION-2B, and DFN-5B. Add CLIP4STR models trained on RBU(6.5M).


## Introduction

<div align="justify">

This is a **third-party implementation** of the paper 
<a href="https://arxiv.org/abs/2305.14014">
CLIP4STR: A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model.
</a>
<div align=center>
  <img src="misc/overall.png" style="zoom:100%"/></pr>
  
  The framework of CLIP4STR. It has a visual branch and a cross-modal branch. The cross-modal branch
refines the prediction of the visual branch for the final output. The text encoder is partially frozen.
</div>

CLIP4STR aims to build a scene text recognizer with the pre-trained vision-language model. In this re-implementation,
we try to reproduce the performance of the original paper and evaluate the effectiveness of pre-trained VL models in the STR area.


## Installation

### Prepare data

First of all, you need to download the STR dataset. 

- We recommend you follow the instructions of [PARSeq](https://github.com/baudm/parseq) at its [parseq/Datasets.md](https://github.com/baudm/parseq/blob/main/Datasets.md) . 
The gdrive links are [gdrive-link1](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) and [gdrive-link2](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) from PARSeq.

- For convenient, you can also download the STR dataset with real training images at [BaiduYunPan str_dataset](https://pan.baidu.com/s/1DY8zYYQ9EHi3_P9pY46DdQ?pwd=hpvw).

- For the RBU(6.5M) training dataset, it is a combination of [the above STR dataset] + [val data of benchmarks (SVT, IIIT5K, IC13, IC15)] + [[Union14M_L_lmdb_format](https://1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ)]. For convenient, you can also download at [BaiduYunPan str_dataset_ub](https://pan.baidu.com/s/1h8I1sUlhR9_YkD7YnEQaEA?pwd=cn67).


- weights of CLIP pre-trained models:
    - [CLIP-ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
    - [CLIP-ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)
    - [CLIP-ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)
    - [OpenCLIP-ViT-B-16-DataComp-XL-s13B-b90K.bin](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/blob/main/open_clip_pytorch_model.bin)
    - [OpenCLIP-ViT-L-14-DataComp-XL-s13B-b90K.bin](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/blob/main/open_clip_pytorch_model.bin)
    - [OpenCLIP-ViT-H-14-laion2B-s32B-b79K.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
    - [appleDFN5B-CLIP-ViT-H-14.bin](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/blob/main/open_clip_pytorch_model.bin)
      - For models from huggingface.co, you should rename them as the shown names.

Generally, directories are organized as follows:
```
${ABSOLUTE_ROOT}
├── dataset
│   │
│   ├── str_dataset_ub
│   └── str_dataset           
│       ├── train
│       │   ├── real
│       │   └── synth
│       ├── val     
│       └── test
│
├── code              
│   │
│   └── clip4str
│
├── output (save the output of the program)
│
│
├── pretrained
│   └── clip (download the CLIP pre-trained weights and put them here)
│       └── ViT-B-16.pt
│
...
```

### Dependency

Requires `Python >= 3.8` and `PyTorch >= 1.12`.
The following commands are tested on a Linux machine with CUDA Driver Version `525.105.17` and CUDA Version `11.3`.
```
conda create --name clip4str python=3.8.5
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
pip install -r requirements.txt 
```

If you meet problems in continual training of an intermediate checkpoint, try to upgrade your PyTorch
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```


## Results

### CLIP4STR pre-trained on OpenAI WIT-400M
`CLIP4STR-B` means using the `CLIP-ViT-B/16` as the backbone, and `CLIP4STR-L` means using the `CLIP-ViT-L/14` as the backbone.


| Method     | Train data | IIIT5K | SVT   | IC13  | IC15  | IC15  | SVTP  | CUTE  | HOST  | WOST  |
|------------|------------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
|            |            | 3,000  | 647   | 1,015 | 1,811 | 2,077 | 645   | 288   | 2,416 | 2,416 | 
| CLIP4STR-B  | MJ+ST      | 97.70  | 95.36 | 96.06 | 87.47 | 84.02 | 91.47 | 94.44 | 80.01 | 86.75 | 
| CLIP4STR-L  | MJ+ST      | 97.57  | 95.36 | 96.75 | 88.02 | 84.40 | 91.78 | 94.44 | 81.08 | 87.38 |
| CLIP4STR-B  | Real(3.3M) | 99.20  | 98.30 | 98.23 | 91.44 | 90.61 | 96.90 | 99.65 | 77.36 | 87.87 | 
| CLIP4STR-L  | Real(3.3M) | 99.43  | 98.15 | 98.52 | 91.66 | 91.14 | 97.36 | 98.96 | 79.22 | 89.07 | 


| Method     | Train data | COCO  | ArT    | Uber   | | Checkpoint |
|------------|------------|-------|--------|--------|-|--------|
|            |            | 9,825 | 35,149 | 80,551 | | |
|CLIP4STR-B  | MJ+ST      | 66.69 | 72.82  | 43.52  | | [a5e3386222](https://github.com/VamosC/CLIP4STR/releases/download/1.0.0/clip4str_base16x16_synth_a5e3386222.ckpt) |
|CLIP4STR-L  | MJ+ST      | 67.45 | 73.48  | 44.59  | | [3544c362f0](https://zenodo.org/record/8369439) |
|CLIP4STR-B  | Real(3.3M) | 80.80 | 85.74  | 86.70  | | [d70bde1f2d](https://github.com/VamosC/CLIP4STR/releases/download/1.0.0/clip4str_base16x16_d70bde1f2d.ckpt) |
|CLIP4STR-L  | Real(3.3M) | 81.97 | 85.83  | 87.36  | | [f125500adc](https://zenodo.org/record/8369439) |


### CLIP4STR pre-trained on DataComp-1B, LAION-2B, and DFN-5B

All models are trained on RBU(6.5M).

| Method     | Pre-train | Train | IIIT5K | SVT   | IC13  | IC15  | IC15  | SVTP  | CUTE  | HOST  | WOST  |
|------------|------------|------------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
|            |            |            | 3,000  | 647   | 1,015 | 1,811 | 2,077 | 645   | 288   | 2,416 | 2,416 | 
| CLIP4STR-B  | DC-1B | RBU | 99.5  | 98.3 | 98.6 | 91.4 | 91.1 | 98.0 | 99.0 | 79.3 | 88.8 | 
| CLIP4STR-L  | DC-1B | RBU | 99.6  | 98.6 | 99.0 | 91.9 | 91.4 | 98.1 | 99.7 | 81.1 | 90.6 |
| CLIP4STR-H  | LAION-2B | RBU | 99.7 | 98.6 | 98.9 |	91.6	| 91.1	| 98.5	| 99.7	| 80.6	| 90.0 |
| CLIP4STR-H  | DFN-5B | RBU | 99.5	| 99.1	| 98.9	| 91.7	| 91.0	| 98.0	| 99.0	| 82.6	| 90.9 |


| Method     | Pre-train | Train | COCO   | ArT    | Uber   | log | Checkpoint |
|------------|----------------|------------|--------|--------|--------|-----|--------|
|            |                |            | 9,825  | 35,149 | 80,551 |     |        |
|CLIP4STR-B  | DC-1B    | RBU  | 81.3  | 85.8  | 92.1  | [6e9fe947ac_log](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_base_6e9fe947ac_log.txt)    | [6e9fe947ac](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_base_6e9fe947ac.pt), [BaiduYun](https://pan.baidu.com/s/1KCmM4K16nslPoZbUPxIfnQ?pwd=g6z6) |
|CLIP4STR-L  | DC-1B    | RBU  | 82.7  |86.4  | 92.2  | [3c9d881b88_log](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_large_3c9d881b88_log.txt)    | [3c9d881b88](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_large_3c9d881b88.pt), [BaiduYun](https://pan.baidu.com/s/1V1Wd114MUhszdPwQyD85DA?pwd=4q74) |
|CLIP4STR-H  | LAION-2B       | RBU  | 82.5  | 86.2  | 91.2 | [5eef9f86e2_log](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_huge_5eef9f86e2_log.txt)    | [5eef9f86e2](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_huge_5eef9f86e2.pt), [BaiduYun](https://pan.baidu.com/s/1RFOdzHFEeDOs5EC2F3u3Ng?pwd=ex5w) |
|CLIP4STR-H  | DFN-5B         | RBU  | 83.0  | 86.4 | 91.7  | [3e942729b1_log](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_huge_3e942729b1_log.txt)    | [3e942729b1](https://huggingface.co/mzhaoshuai/CLIP4STR/blob/main/clip4str_huge_3e942729b1.pt), [BaiduYun](https://pan.baidu.com/s/1duQnOsKZcOF3oiGlsdkBHw?pwd=e2ty) |


## Training

- Before training, you should set the path properly. Find the `/PUT/YOUR/PATH/HERE` in `configs`, `scripts`, `strhub/vl_str`, and `strhub/str_adapter`. For example, the `/PUT/YOUR/PATH/HERE` in the `configs/main.yaml`. Then replace it with your own path. A global searching and replacement is recommended.


For CLIP4STR with `CLIP-ViT-B`, refer to
```
bash scripts/vl4str_base.sh
```
8 NVIDIA GPUs with more than 24GB memory (per GPU) are required.
For users with limited GPUs,
you can change `trainer.gpus=A`, `trainer.accumulate_grad_batches=B`, and `model.batch_size=C` under the condition `A * B * C = 1024` in the bash scripts.
Do not modify the code, the `PyTorch Lightning` will handle the left.


For CLIP4STR with `CLIP-ViT-L`, refer to
```
bash scripts/vl4str_large.sh
```

We also provide the training script of `CLIP4STR + Adapter` described in the original paper,
```
bash scripts/str_adapter.sh
```


## Inference


```
bash test.sh {gpu_id} {subpath_of_ckpt}
```
For example,
```
bash scripts/test.sh 0 clip4str_base16x16_d70bde1f2d.ckpt
```

If you want to read characters from an image, try:
```
bash read.sh {gpu_id} {subpath_of_ckpt} {image_folder_path}
```
For example,
```
bash scripts/read.sh 0 clip4str_base16x16_d70bde1f2d.ckpt misc/test_images

Output:
image_1576.jpeg: Chicken
```

## FAQs

### CLIP4STR in other languages

Please check https://github.com/VamosC/CLIP4STR/issues/1, https://github.com/VamosC/CLIP4STR/issues/20, https://github.com/VamosC/CLIP4STR/issues/23.

If you have implemented CLIP4STR in other languages, it would be great if you could add a link to your repo here.

### Code for producing attention map in CLIP4STR paper

Please check https://github.com/VamosC/CLIP4STR/issues/6.


## Citations
```
@article{zhao2024clip4str,
  author={Zhao, Shuai and Quan, Ruijie and Zhu, Linchao and Yang, Yi},
  journal={IEEE Transactions on Image Processing}, 
  title={CLIP4STR: A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model}, 
  year={2024},
  pages={1-1},
  doi={10.1109/TIP.2024.3512354}}
```

## Acknowledgements

<!--ts-->
* [baudm/parseq](https://github.com/baudm/parseq)
* [openai/CLIP](https://github.com/openai/CLIP)
* [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* [huggingface/transformers](https://github.com/huggingface/transformers)
* [large-ocr-model/large-ocr-model.github.io](https://github.com/large-ocr-model/large-ocr-model.github.io)
* [Mountchicken/Union14M](https://github.com/Mountchicken/Union14M)
* [mzhaoshuai/CenterCLIP](https://github.com/mzhaoshuai/CenterCLIP)
* [VamosC/CoLearning-meet-StitchUp](https://github.com/VamosC/CoLearning-meet-StitchUp)
* [VamosC/CapHuman](https://github.com/VamosC/CapHuman)
* Dr. Xiaohan Wang from Stanford University.
<!--te-->
