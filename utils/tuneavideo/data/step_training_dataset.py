# https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/data/dataset.py

import decord
decord.bridge.set_bridge('torch')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class StepTrainingSepDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None
        self.uncond_prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        if 'mp4' not in self.video_path:
            self.images = []
            images_list = [file for file in os.listdir(self.video_path) if file.endswith('jpg') or file.endswith('png')]
            for file in sorted(images_list, key=lambda x: int(x[:-4])):
                self.images.append(np.asarray(Image.open(os.path.join(self.video_path, file)).convert('RGB').resize((self.width, self.height))))
        self.images = np.stack(self.images)

        if "masks" in os.listdir(self.video_path):
            self.video_masks = []
            mask_path = os.path.join(self.video_path, "masks")
            for file in sorted(os.listdir(mask_path), key=lambda x: int(x[:-4])):
                self.video_masks.append(np.asarray(Image.open(os.path.join(mask_path, file)).convert('L').resize((self.width, self.height))))
        self.video_masks = np.stack(self.video_masks)

        # Motion index and initializer
        self.s1_training = True
        self.s1_prompt_ids = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Load and sample video frames
        if 'mp4' in self.video_path:
            vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
            sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
            video = vr.get_batch(sample_index)
        else:
            video = self.images[:self.n_sample_frames]
        video = rearrange(video, "f h w c -> f c h w")

        prompt_ids = self.s1_prompt_ids if self.s1_training else self.prompt_ids
        video_masks = (self.video_masks > 0).astype(np.uint8)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
            "instance_masks": video_masks
        }

        return example
