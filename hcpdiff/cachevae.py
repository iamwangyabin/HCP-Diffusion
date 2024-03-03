import argparse
import math
import os
import time
import warnings
from functools import partial

import diffusers
import hydra
import torch
import torch.utils.checkpoint
# fix checkpoint bug for train part of model
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from omegaconf import OmegaConf

from hcpdiff.ckpt_manager import CkptManagerPKL, CkptManagerSafe
from hcpdiff.data import RatioBucket, DataGroup, get_sampler
from hcpdiff.loggers import LoggerGroup
from hcpdiff.utils.net_utils import get_scheduler, auto_tokenizer_cls, auto_text_encoder_cls, load_emb
from hcpdiff.utils.utils import load_config_with_cli, get_cfg_range, mgcd, format_number

def checkpoint_fix(function, *args, use_reentrant: bool = False, checkpoint_raw=torch.utils.checkpoint.checkpoint, **kwargs):
    return checkpoint_raw(function, *args, use_reentrant=use_reentrant, **kwargs)

torch.utils.checkpoint.checkpoint = checkpoint_fix

class Trainer:
    weight_dtype_map = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}
    ckpt_manager_map = {'torch':CkptManagerPKL, 'safetensors':CkptManagerSafe}

    def __init__(self, cfgs_raw):
        cfgs = hydra.utils.instantiate(cfgs_raw)
        self.cfgs = cfgs

        self.init_context(cfgs_raw)
        self.build_loggers(cfgs_raw)
        self.build_model()
        self.batch_size_list = []
        assert len(cfgs.data)>0, "At least one dataset is need."
        loss_weights = [dataset.keywords['loss_weight'] for name, dataset in cfgs.data.items()]
        self.train_loader_group = DataGroup([self.build_data(dataset) for name, dataset in cfgs.data.items()], loss_weights)

        torch.backends.cuda.matmul.allow_tf32 = cfgs.allow_tf32

        # calculate steps and epochs
        self.steps_per_epoch = len(self.train_loader_group.loader_list[0])
        if self.cfgs.train.train_epochs is not None:
            self.cfgs.train.train_steps = self.cfgs.train.train_epochs*self.steps_per_epoch
        else:
            self.cfgs.train.train_epochs = math.ceil(self.cfgs.train.train_steps/self.steps_per_epoch)


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    def init_context(self, cfgs_raw):
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],  # fix inplace bug in DDP while use data_class
        )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed+self.local_rank)

    def build_loggers(self, cfgs_raw):
        if self.is_local_main_process:
            self.exp_dir = self.cfgs.exp_dir.format(time=time.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(os.path.join(self.exp_dir, 'ckpts/'), exist_ok=True)
            with open(os.path.join(self.exp_dir, 'cfg.yaml'), 'w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(cfgs_raw))
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=self.exp_dir) for builder in self.cfgs.logger])
        else:
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=None) for builder in self.cfgs.logger])

        self.min_log_step = mgcd(*([item.log_step for item in self.loggers.logger_list]))
        image_log_steps = [item.image_log_step for item in self.loggers.logger_list if item.enable_log_image]
        if len(image_log_steps)>0:
            self.min_img_log_step = mgcd(*image_log_steps)
        else:
            self.min_img_log_step = -1

        self.loggers.info(f'world_size: {self.world_size}')
        self.loggers.info(f'accumulation: {self.cfgs.train.gradient_accumulation_steps}')

        if self.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()


    def build_model(self):
        # Load the tokenizer
        if self.cfgs.model.get('tokenizer', None) is not None:
            self.tokenizer = self.cfgs.model.tokenizer
        else:
            tokenizer_cls = auto_tokenizer_cls(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            self.tokenizer = tokenizer_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=self.cfgs.model.revision, use_fast=False,
            )

        self.vae: AutoencoderKL = self.cfgs.model.get('vae', None) or AutoencoderKL.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfgs.model.revision)

        self.weight_dtype = self.weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)
        self.vae_dtype = self.weight_dtype_map.get(self.cfgs.model.get('vae_dtype', None), torch.float32)
        self.vae = self.vae.to(self.device, dtype=self.vae_dtype)



    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop('batch_size')
        cache_latents = data_builder.keywords.pop('cache_latents')
        self.batch_size_list.append(batch_size)

        train_dataset = data_builder(tokenizer=self.tokenizer, tokenizer_repeats=self.cfgs.model.tokenizer_repeats)
        train_dataset.bucket.build(batch_size*self.world_size, file_names=train_dataset.source.get_image_list())
        arb = isinstance(train_dataset.bucket, RatioBucket)
        self.loggers.info(f"len(train_dataset): {len(train_dataset)}")
        return train_dataset, batch_size, arb

    def build_data(self, data_builder: partial) -> torch.utils.data.DataLoader:
        train_dataset, batch_size, arb = self.build_dataset(data_builder)

        # Pytorch Data loader
        train_sampler = get_sampler()(train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=not arb)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=self.cfgs.train.workers,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        return train_loader

    def train(self):
        total_batch_size = sum(self.batch_size_list)*self.world_size*self.cfgs.train.gradient_accumulation_steps

        self.loggers.info("***** Running training *****")
        self.loggers.info(f"  Num batches each epoch = {len(self.train_loader_group.loader_list[0])}")
        self.loggers.info(f"  Num Steps = {self.cfgs.train.train_steps}")
        self.loggers.info(f"  Instantaneous batch size per device = {sum(self.batch_size_list)}")
        self.loggers.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.global_step = 0
        if self.cfgs.train.resume is not None:
            self.global_step = self.cfgs.train.resume.start_step

        for data_list in self.train_loader_group:
            with torch.no_grad():
                names, ls, mask, crop_info = self.train_one_step(data_list)

            tmp = [names, ls, mask, crop_info]
            torch.save(tmp, os.path.join(self.cfgs.cache_path, "latents"+str(names[0])+'.pth'))
            del tmp
            self.global_step += 1
            if self.is_local_main_process:
                if self.global_step%self.min_log_step == 0:
                    self.loggers.log(datas={'Step':{'format':'[{}/{}]', 'data':[self.global_step, self.cfgs.train.train_steps]}}, step=self.global_step)
                    import gc
                    gc.collect()

            if self.global_step>=self.cfgs.train.train_steps:
                break

        self.wait_for_everyone()


    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def get_latents(self, image, dataset):
        if dataset.latents is None:
            latents = self.vae.encode(image.to(dtype=self.vae.dtype)).latent_dist.sample()
            latents = latents*self.vae.config.scaling_factor
        else:
            latents = image  # Cached latents
        return latents

    def train_one_step(self, data_list):
        for idx, data in enumerate(data_list):
            image = data.pop('img').to(self.device, dtype=self.weight_dtype)
            latents = self.get_latents(image, self.train_loader_group.get_dataset(idx))

        return data['img_name'].tolist(), latents.cpu(), data['mask'].cpu(), data['crop_info'].cpu()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = Trainer(conf)
    trainer.train()

