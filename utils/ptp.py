import os
import abc
import math
import torch
import torch.nn.functional as nnf
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils.ptp_utils as ptp_utils
import utils.seq_aligner as seq_aligner

from typing import Optional, Union, Tuple, List, Callable, Dict
from PIL import Image
from einops import rearrange

from utils.ptp_utils import reshape_heads_to_batch_dim, reshape_batch_dim_to_heads

MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# need to adjust sometimes
mask_th = (.3, .3)


class LocalBlend:

    def get_mask(self, maps, alpha, use_pool, x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(2)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[3:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store, step):
        if self.latents is not None:
            x_t[0:1] = self.latents[-(step + 1)]
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            # maps = attention_store["down_cross"][4:8] + attention_store["up_cross"][:6] ############ gated ca
            # maps = [attn[:, :256, :] for attn in attention_store["down_cross"][2:4]] + \
            #        [attn[:, :256, :] for attn in attention_store["up_cross"][:3]] ############ gated sa
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=2)
            mask = self.get_mask(maps, self.alpha_layers, True, x_t)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, substruct_words=None, start_blend=0.2,
                 th=(.3, .3), latents=None):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th
        self.latents = latents


class NoiseBlend(LocalBlend):

    def __call__(self, x_t, attention_store, step):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=2)
        mask = self.get_mask(maps, self.alpha_layers, True, x_t)
        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
            mask = mask * maps_sub
        mask = mask.float()
        mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
        transform = T.GaussianBlur(kernel_size=3, sigma=3)
        x_curr = transform(x_t[1])
        # x_curr = x_t[1] + (0.1 ** 0.5) * torch.randn_like(x_t[1])
        x_t[1] = x_curr * mask[1] + x_t[1] * (1 - mask[1])
        return x_t


class SDEditNoise(LocalBlend):
    def extract(self, a, t, x_shape):
        """
        Extract coefficients from a based on t and reshape to make it
        broadcastable with x_shape.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
        assert out.shape == (bs,)
        out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
        return out

    def image_editing_denoising_step_flexible_mask(self, x, t, *, model, logvar, betas):
        """
        Sample from p(x_{t-1} | x_t)
        """
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)

        model_output = model(x, t)
        weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
        mean = self.extract(1 / torch.sqrt(alphas), t, x.shape) * (x - self.extract(weighted_score, t, x.shape) * model_output)

        logvar = self.extract(logvar, t, x.shape)
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        sample = sample.float()
        return sample

    def __call__(self, x_t, attention_store, step):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=2)
        mask = self.get_mask(maps, self.alpha_layers, True, x_t)
        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
            mask = mask * maps_sub
        mask = mask.float()
        mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
        e = torch.randn_like(self.x0)
        a = (1 - self.betas).cumprod(dim=0)
        x = self.x0 * a[self.total_noise_levels - 1].sqrt() + e * (1.0 - a[self.total_noise_levels - 1]).sqrt()
        for i in reversed(range(self.total_noise_levels)):
            t = (torch.ones(2) * i).to(device)
            x_ = self.image_editing_denoising_step_flexible_mask(
                x, t=t, model=self.unet, logvar=self.logvar, betas=self.betas
            )
            x = self.x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
            x[:, (mask != 1.)] = x_[:, (mask != 1.)]
        return x_t

    def __init__(self, betas, x0, total_noise_levels, unet, logvar,
                 prompts: List[str], words: [List[List[str]]], tokenizer, substruct_words=None, start_blend=0.2,
                 th=(.3, .3), latents=None):
        super(SDEditNoise, self).__init__(prompts, words, tokenizer, substruct_words, start_blend, th, latents)
        self.betas = betas
        self.x0 = x0
        self.total_noise_levels = total_noise_levels
        self.unet = unet
        self.logvar = logvar


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, store_only)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_frame": [], "mid_frame": [], "up_frame": [],
                "down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        key = f"{place_in_unet}_{'cross' if is_cross else 'frame' if store_only else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        if self.cur_step != 50:
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.step_store}  #self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionStore__(AttentionControl):

    def step_callback(self, x_t):
        # if self.local_blend is not None and self.cur_step > 25:
        #     x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        # if self.noise_blend is not None and 5 < self.cur_step < 10:
        #     x_t = self.noise_blend(x_t, self.attention_store, self.cur_step)
        x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    @staticmethod
    def get_empty_store():
        return {"down_frame": [], "mid_frame": [], "up_frame": [],
                "down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        key = f"{place_in_unet}_{'cross' if is_cross else 'frame' if store_only else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        if self.cur_step != 50:
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore__, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, local_blend=None, noise_blend=None):
        super(AttentionStore__, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.local_blend = local_blend
        self.noise_blend = noise_blend


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, store_only)
        if store_only:
            return attn
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]], tokenizer,
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, tokenizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]],
                  tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, tokenizer, blend_words=None, equilizer_params=None,
                    mask_th=(.3, .3)) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer, th=mask_th)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, tokenizer=tokenizer, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, tokenizer=tokenizer, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, tokenizer=tokenizer,
                                       local_blend=lb, controller=controller)
    return controller


def load_512_seq(image_path, left=0, right=0, top=0, bottom=0, n_sample_frame=8, sampling_rate=1, start_frame=0):
    images = []
    for file in sorted(os.listdir(image_path)):
        if file.endswith('jpg') or file.endswith('png'):
            images.append(file)
    n_images = len(images)
    sequence_length = (n_sample_frame - 1) * sampling_rate + 1
    if n_images < sequence_length:
        raise ValueError
    frames = []
    for index in range(start_frame, start_frame + n_sample_frame):
        p = os.path.join(image_path, images[index])
        image = np.array(Image.open(p).convert("RGB"))
        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h - bottom, left:w - right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((512, 512)))
        frames.append(image)
    return np.stack(frames)


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], tokenizer, prompts,
                         select=0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_temporal_attention(attention_store: AttentionStore, res: int, from_where: List[str], prompts, video_len,
                            num_heads=8, select=0, image_path=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_self"]:
            if item.shape[0] // (len(prompts) * num_heads) == num_pixels:
                cross_maps = item.reshape(len(prompts), res, res, num_heads, video_len, video_len)[select]
                out.append(cross_maps.permute(2, 0, 1, 3, 4))
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    out = out.cpu()

    grid_ratio = 0.1
    start_ratio = (0, 0.5)
    end_ratio = (0.7, 1)
    if isinstance(image_path, Image.Image):
        pil_img = image_path
    elif isinstance(image_path, str):
        image_gt = load_512_seq(image_path)
        pil_img = Image.fromarray(image_gt[0])
    else:
        raise NotImplementedError
    fig, ax = plt.subplots()
    ax.imshow(pil_img)
    for i in range(int((end_ratio[0] - start_ratio[0]) // grid_ratio) + 1):
        for j in range(int((end_ratio[1] - start_ratio[1]) // grid_ratio) + 1):
            rect = patches.Rectangle(
                (int(pil_img.size[1] * start_ratio[1] + pil_img.size[1] * grid_ratio * j),
                 int(pil_img.size[0] * start_ratio[0] + pil_img.size[0] * grid_ratio * i)),
                10, 10, linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
    plt.axis('off')
    plt.show()

    images = []
    for i in range(int((end_ratio[0] - start_ratio[0]) // grid_ratio) + 1):
        for j in range(int((end_ratio[1] - start_ratio[1]) // grid_ratio) + 1):
            image = out[int(out.shape[1] * start_ratio[0] + out.shape[1] * grid_ratio * i),
                        int(out.shape[2] * start_ratio[1] + out.shape[2] * grid_ratio * j)]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            #np.array(Image.fromarray(image).resize((256, 256)))
            image = np.concatenate([np.ones_like(image[:, 0:1]) * 255., image, np.ones_like(image[:, 0:1]) * 255.], axis=1)
            images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), num_rows=i)


def show_frame_attention(attention_store: AttentionStore, res: int, from_where: List[str], prompts, video_len,
                         select=0, image_path=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_frame"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), video_len, -1, num_pixels, num_pixels)[select]
                out.append(cross_maps.permute(1, 0, 2, 3))
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    out = out.cpu()

    # Visualize shifted distance
    row_max = torch.argmax(out, dim=-1) // res  # (res, res, frame)
    col_max = torch.argmax(out, dim=-1) % res
    row_ind = torch.arange(0, res).repeat_interleave(res)[None, :]
    col_ind = torch.arange(0, res).repeat(res)[None, :]
    dist = torch.sqrt((row_max - row_ind) ** 2 + (col_max - col_ind) ** 2)

    images = []
    for i in range(video_len):  # F_i frame
        image = dist[i].view(res, res)
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))

    # Visualize spatial attention map per location
    # grid_size = 4
    # if isinstance(image_path, Image.Image):
    #     pil_img = image_path
    # elif isinstance(image_path, str):
    #     image_gt = load_512_seq(image_path)
    #     pil_img = Image.fromarray(image_gt[0])
    # else:
    #     raise NotImplementedError
    # fig, ax = plt.subplots()
    # ax.imshow(pil_img)
    # for r in range(1, grid_size):
    #     for c in range(1, grid_size):
    #         rect = patches.Rectangle(
    #             (r * (pil_img.size[0] // grid_size), c * (pil_img.size[1] // grid_size)),
    #             10, 10, linewidth=1, edgecolor='r', facecolor='none'
    #         )
    #         ax.add_patch(rect)
    # plt.axis('off')
    # plt.show()
    #
    # for i in range(len(out)):  # F_i frame
    #     images = []
    #     for r in range(1, grid_size):  # location in F_0 frame
    #         for c in range(1, grid_size):
    #             image = out[i, r * (out.shape[1] // grid_size), c * (out.shape[2] // grid_size)].view(res, res)
    #             image = 255 * image / image.max()
    #             image = image.unsqueeze(-1).expand(*image.shape, 3)
    #             image = image.numpy().astype(np.uint8)
    #             image = np.array(Image.fromarray(image).resize((256, 256)))
    #             images.append(image)
    #     ptp_utils.view_images(np.stack(images, axis=0))


def show_cross_attention_per_frame(attention_store: AttentionStore, res: int, from_where: List[str], tokenizer, prompts,
                                   video_len, select=0, select_words=[], save_dir=None):
    # Get cross attention map averaged across heads
    out = []
    attention_maps = attention_store.get_average_attention()
    # attention_maps = {key: [item for item in attention_store.step_store[key]] for key in attention_store.step_store}
    for location in from_where:
        for item in attention_maps[f"{location}_cross"]:
            if res == -1:
                item_res, seq_len = int(item.shape[1] ** 0.5), item.shape[-1]
                cross_maps = item.reshape(len(prompts), video_len, -1, item_res, item_res, seq_len)[select]
                cross_maps = cross_maps.permute(0, 1, 4, 2, 3).reshape(video_len, -1, item_res, item_res)
                cross_maps = nnf.interpolate(cross_maps, size=(64, 64), mode='bilinear')
                cross_maps = cross_maps.reshape(video_len, -1, seq_len, 64, 64).permute(0, 1, 3, 4, 2)
                out.append(cross_maps)
            elif item.shape[1] == res ** 2:
                cross_maps = item.reshape(len(prompts), video_len, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode

    if save_dir is not None:
        save_dir = os.path.join(save_dir, prompts[select].replace(" ", "_"))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # [Visualize] How much does `the word` influence each pixel in a single frame (sum across pixels = 1)
    for word in select_words:
        if word not in prompts[select]:
            continue
        w = prompts[select].split(' ').index(word) + 1
        images = []
        for f in range(len(out)):
            image = out[f, :, :, w]

            cmap = plt.get_cmap('jet')
            image = image / image.max()
            plt.matshow(image.detach().cpu().numpy(), cmap=cmap)
            # plt.colorbar(aspect=10)
            plt.clim(0, 1)
            plt.axis('off')
            plt.show()

            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.detach().cpu().numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[w])))
            images.append(image)

        ptp_utils.view_images(np.stack(images, axis=0), set_title=f"{select}-{word}", save_dir=save_dir)

    # [Visualize] How much does `the pixel` influenced by words in prompts (sum across words = 1)
    images = []
    attention_scale = torch.sum(out[0, 0, 0, :])
    for f in range(len(out)):
        # exclude <bos>, <eos> and re-weight semantic tokens
        tkns = len(tokens) - 1
        reweighed_image = out[f, :, :, 1:tkns] / torch.sum(out[f, :, :, 1:tkns], dim=-1, keepdim=True)
        for i in range(reweighed_image.shape[-1]):  # range(len(tokens)):
            image = reweighed_image[:, :, i]  # out[f, :, :, i] / attention_scale
            image = 255 * image
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.detach().cpu().numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, f"{torch.sum(reweighed_image[:, :, i]) / reweighed_image.shape[0] ** 2:.4f}"
            )  # decoder(int(tokens[i]))) / decoder(int(tokens[i + 1])))
            images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), num_rows=len(out), set_title=f"{select}-ca_map", save_dir=save_dir)
    return torch.sum(out[:, :, :, reweighed_image.shape[-1] - 1]) / (attention_scale * len(out) * res ** 2)


def register_attention_control_exp(model, controller):

    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = reshape_heads_to_batch_dim(q, head_size=h)
            k = reshape_heads_to_batch_dim(k, head_size=h)
            v = reshape_heads_to_batch_dim(v, head_size=h)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = torch.exp(sim - torch.max(sim)) / torch.sum(torch.exp(sim - torch.max(sim)), axis=-1).unsqueeze(-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(out, head_size=h)
            return to_out(out)

        return forward

    def frame_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
            batch_size, sequence_length, _ = hidden_states.shape
            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            query = self.head_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            former_frame_index = torch.arange(video_length) - 1
            former_frame_index[0] = 0

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)  #key[:, [0] * video_length]
            key = rearrange(key, "b f d c -> (b f) d c")

            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)  #value[:, [0] * video_length]
            value = rearrange(value, "b f d c -> (b f) d c")

            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            sim = torch.einsum('b i d, b j d -> b i j', query, key) / math.sqrt(query.size(-1))
            attn = sim.softmax(dim=-1)
            attn = controller(attn, False, place_in_unet, store_only=True)

            hidden_states = torch.einsum('b i j, b j d -> b i d', attn, value)
            hidden_states = hidden_states.reshape(batch_size, self.heads, -1, dim // self.heads)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, dim)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        return forward

    def register_recr(name_, net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention' and name_ == "attn2":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        # elif net_.__class__.__name__ == 'FrameAttention':
        #     net_.forward = frame_forward(net_, place_in_unet)
        #     return count + 1
        elif hasattr(net_, 'children') and 'Gated' not in net_.__class__.__name__:
            for name__, net__ in net_.named_children():
                count = register_recr(name__, net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children() if hasattr(model, 'unet') else model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[0], net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[0], net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[0], net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

