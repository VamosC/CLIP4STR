#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import hydra
import torch
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import create_model
from strhub.dist_utils import copy_remote, is_main_process


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpus = config.trainer.get('gpus', 0)
        if gpus:
            # Use mixed-precision training
            config.trainer.precision = 16
        if gpus > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=getattr(config.model, "find_unused_parameters", False),
                                            gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= gpus
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= gpus

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # print config
    rank_zero_info(OmegaConf.to_yaml(config))

    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        model: BaseSystem = create_model(config.pretrained, True)
    else:
        model: BaseSystem = hydra.utils.instantiate(config.model)
    
    rank_zero_info(summarize(model, max_depth=1 if model.hparams.name.startswith('parseq') else 2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)
    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True,
                                    filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
                                    every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint]
    if getattr(config, 'swa', False):
        # set swa lrs
        swa_epoch_start = 0.8
        lr_scale = math.sqrt(torch.cuda.device_count()) * config.data.batch_size / 256.
        lr = lr_scale * config.model.lr
        if "clip" in config.model.name:
            swa_lrs = [lr, lr, config.model.coef_lr * lr, config.model.coef_lr * lr]
        else:
            swa_lrs = [lr,]

        swa_lrs = [x * (1 - swa_epoch_start) for x in swa_lrs]
        swa = StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=swa_epoch_start)
        callbacks.append(swa)

    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
            str(Path(config.ckpt_path).parents[1].absolute())
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               accumulate_grad_batches=config.trainer.accumulate_grad_batches,
                                               callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # copy data and perform test
    torch.distributed.barrier()
    if is_main_process():
        copy_remote(cwd, config.data.output_url)
        test_call(cwd, config.data.root_dir, config.model.code_path)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@rank_zero_only
def test_call(cwd, data_dir, code_path=None):
    file = os.path.join(code_path, 'test.py')
    assert os.path.exists(file)
    print("The execute file is {}".format(file))
    ckpts = [x for x in os.listdir(os.path.join(cwd, 'checkpoints')) if 'val' in x]
    val_acc = [float(x.split('-')[-2].split('=')[-1]) for x in ckpts]

    best_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), ckpts[val_acc.index(max(val_acc))])
    print("The best ckpt is {}".format(best_ckpt))
    best_epoch = int(best_ckpt.split('/')[-1].split('-')[0].split('=')[-1])
    print('The val accuracy is best {}-{}e'.format(max(val_acc), best_epoch))

    # test best
    # print("\n Test results with the best checkpoint")
    # os.system("python {} {} --data_root {} --new".format(file, best_ckpt, data_dir))
    # test last

    print("\n Test results with the last checkpoint")
    last_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), "last.ckpt")
    os.system("python {} {} --data_root {} --new".format(file, last_ckpt, data_dir))

    # test last with refinement
    # print("\n Test results with the last checkpoint")
    # last_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), "last.ckpt")
    # os.system("python {} {} --data_root {} --new --clip_refine".format(file, last_ckpt, data_dir))


if __name__ == '__main__':
    main()
