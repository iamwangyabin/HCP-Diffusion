import argparse
import os
import random
from typing import List

import hydra
import torch
from hcpdiff.utils.utils import load_config_with_cli, prepare_seed, is_list
from torch.cuda.amp import autocast
from diffusers import DiffusionPipeline


class Visualizer:
    def __init__(self, cfgs):
        self.cfgs_raw = cfgs
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.pipe = DiffusionPipeline.from_pretrained(self.cfgs.pretrained_model, safety_checker=None, requires_safety_checker=False, resume_download=True,
                                                torch_dtype=torch.float16)

        # from hcpdiff.ckpt_manager import CkptManagerSafe
        # manger = CkptManagerSafe()
        # unet = manger.load_ckpt(cfgs.resume_ckpt_path.unet)
        # self.pipe.unet.load_state_dict(unet['base'], True)
        # te = manger.load_ckpt(cfgs.resume_ckpt_path.TE)
        # clip_bigG_weights = {}
        # clip_B_weights = {}
        # for key in te['base'].keys():
        #     if key.startswith('clip_bigG.'):
        #         new_key = key[len('clip_bigG.'):]
        #         clip_bigG_weights[new_key] = te['base'][key]
        #     elif key.startswith('clip_B.'):
        #         new_key = key[len('clip_B.'):]
        #         clip_B_weights[new_key] = te['base'][key]
        # self.pipe.text_encoder.load_state_dict(clip_B_weights, False)
        # self.pipe.text_encoder_2.load_state_dict(clip_bigG_weights, False)
        # print("load model")

        self.pipe.to("cuda")

    @torch.no_grad()
    def vis_images(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        G = prepare_seed(seeds or [None]*len(prompt))
        images = self.pipe(prompt, generator=G, **self.cfgs.infer_args).images
        return images

    def save_images(self, images, prompt, negative_prompt='', seeds: List[int] = None):
        for interface in self.cfgs.interface:
            interface.on_infer_finish(images, prompt, negative_prompt, self.cfgs_raw, seeds=seeds)

    def vis_to_dir(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]
        images = self.vis_images(prompt, negative_prompt, seeds=seeds, **kwargs)
        self.save_images(images, prompt, negative_prompt, seeds=seeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg

    if cfgs.seed is not None:
        if is_list(cfgs.seed):
            assert len(cfgs.seed) == cfgs.num*cfgs.bs, 'seed list length should be equal to num*bs'
            seeds = list(cfgs.seed)
        else:
            seeds = list(range(cfgs.seed, cfgs.seed+cfgs.num*cfgs.bs))
    else:
        seeds = [None]*(cfgs.num*cfgs.bs)

    viser = Visualizer(cfgs)
    for i in range(cfgs.num):
        prompt = cfgs.prompt[i*cfgs.bs:(i+1)*cfgs.bs] if is_list(cfgs.prompt) else [cfgs.prompt]*cfgs.bs
        negative_prompt = cfgs.neg_prompt[i*cfgs.bs:(i+1)*cfgs.bs] if is_list(cfgs.neg_prompt) else [cfgs.neg_prompt]*cfgs.bs
        viser.vis_to_dir(prompt=prompt, negative_prompt=negative_prompt,
                         seeds=seeds[i*cfgs.bs:(i+1)*cfgs.bs], save_cfg=cfgs.save.save_cfg, **cfgs.infer_args)
