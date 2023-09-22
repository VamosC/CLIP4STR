#!/bin/bash

### LadderSideAdapter
# python ../train.py +experiment=str_adapter model=str_adapter charset=94_full dataset=real \
#                     model.lr=8.4e-5 \
#                     model.batch_size=256 \
#                     trainer.max_epochs=20 trainer.gpus=4 \
#                     model.adapter_type=ladder model.block_ids=[1,3,5,7,9,11]


### LadderSideAdapter with pruned weights
# python ../train.py +experiment=str_adapter model=str_adapter charset=94_full dataset=real \
#                     model.lr=8e-4 \
#                     model.batch_size=256 \
#                     trainer.max_epochs=20 trainer.gpus=4 \
#                     model.adapter_type=ladder_pruning model.block_ids=[1,3,5,7,9,11] \
#                     model.prune_reduction=4

python ../train.py +experiment=str_adapter model=str_adapter charset=94_full dataset=real \
                    model.lr=8e-4 \
                    model.batch_size=512 \
                    trainer.max_epochs=16 trainer.gpus=2 \
                    model.adapter_type=ladder_pruning model.block_ids=[1,3,5,7,9,11] \
                    model.prune_reduction=8
