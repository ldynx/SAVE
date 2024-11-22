import types
import os
import re
import math
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available

from utils.ptp import AttentionControl, load_512_seq
from utils.ptp_utils import reshape_heads_to_batch_dim, reshape_batch_dim_to_heads

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class LayerAttentionStore(AttentionControl):

    def get_empty_store(self):
        return {layer: [] for layer in self.unet_layer_names}

    def forward(self, attn, type: bool, name: str, store_only: bool = False):
        layer_index = [i for i, layer in enumerate(self.unet_layer_names) if
                       name.startswith(layer.rsplit('.', 1)[0]) and type == layer.rsplit('.', 1)[1]]
        if len(layer_index) > 0:
            if type == "frame":
                # Store frame attention sum to save memory
                num_noise_latents = attn.shape[1]
                if len(self.step_store[self.unet_layer_names[layer_index[0]]]) == 0:
                    self.step_store[self.unet_layer_names[layer_index[0]]] = attn[:, :, :num_noise_latents].clone()
                else:
                    self.step_store[self.unet_layer_names[layer_index[0]]] += attn[:, :, :num_noise_latents].clone()
            else:
                self.step_store[self.unet_layer_names[layer_index[0]]].append(attn)
        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str, store_only: bool = False):
        """
        Overrides `AttentionControl` to store full batch attentions.
        """
        if self.freeze_store:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet, store_only)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # self.between_steps()
        return attn

    def reset(self):
        super(LayerAttentionStore, self).reset()
        self.step_store = self.get_empty_store()

    def freeze_while_backprop(self):
        self.freeze_store = True

    def delete_after_backprop(self):
        self.step_store = self.get_empty_store()
        self.freeze_store = False

    def __init__(self, unet_layer_names):
        super(LayerAttentionStore, self).__init__()
        self.unet_layer_names = unet_layer_names
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.freeze_store = False


def register_layer_attention_store(model, controller, unet_layer_names=None):

    def ca_forward(self, name):
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
            attn = controller(attn, "cross" if is_cross else "temp", name)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(out, head_size=h)
            return to_out(out)

        return forward

    def frame_forward(self, name):
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

            # sim = torch.einsum('b i d, b j d -> b i j', query, key) / math.sqrt(query.size(-1))
            # attn = sim.softmax(dim=-1)
            # attn = controller(attn, "frame", name)
            #
            # hidden_states = torch.einsum('b i j, b j d -> b i d', attn, value)
            # hidden_states = hidden_states.reshape(batch_size, self.heads, -1, dim // self.heads)
            # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, dim)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, "frame", name)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        return forward

    def register_recr(net_, count, name):
        if name.endswith('attn2'):
            net_.forward = ca_forward(net_, name.rsplit('.', 1)[0])
            return count + 1
        elif name.endswith('attn1'):
            net_.forward = frame_forward(net_, name.rsplit('.', 1)[0])
            return count + 1
        elif hasattr(net_, 'children') and 'Gated' not in net_.__class__.__name__:
            for name__, net__ in net_.named_children():
                count = register_recr(net__, count, f"{name}.{name__}")
        return count

    def register_recr_part(net_, count, name):
        if any([name.startswith(layer.rsplit('.', 1)[0]) and
                name.endswith(layer.rsplit('.', 1)[-1].replace('cross', 'attn2').replace('frame', 'attn1'))
                for layer in unet_layer_names]):
            if name.endswith('attn2'):
                net_.forward = ca_forward(net_, name.rsplit('.', 1)[0])
                return count + 1
            elif name.endswith('attn1'):
                net_.forward = frame_forward(net_, name.rsplit('.', 1)[0])
                return count + 1
        if hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                count = register_recr_part(net__, count, f"{name}.{name__}" if len(name) > 0 else f"{name__}")
        return count

    if unet_layer_names is None:
        cross_att_count = register_recr(model, 0, '')
    else:
        cross_att_count = register_recr_part(model, 0, '')
    controller.num_att_layers = cross_att_count


def unet_store_attention_scores(unet, frame_attention_scores, cross_attention_scores, layers=5):
    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_frame_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            frame_attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    def make_new_get_cross_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            cross_attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn1" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            # if isinstance(module.processor, AttnProcessor2_0):
            #     module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_frame_attention_scores_fn(name), module
            )
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            # if isinstance(module.processor, AttnProcessor2_0):
            #     module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_cross_attention_scores_fn(name), module
            )

    return unet


def get_object_localization_loss_for_one_layer(cross_attention_scores, motion_map, object_token_idx, controller, tag):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b = motion_map.shape[0]
    h = bxh // b
    scale = h ** -0.5
    size = int(num_noise_latents ** 0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    motion_map = F.interpolate(motion_map.unsqueeze(1), size=(size, size), mode="bilinear", antialias=True)
    motion_map = motion_map.view(b, -1)  # (b, max_num_objects, num_noise_latents)
    cross_attention_scores = cross_attention_scores.view(b, h, num_noise_latents, num_text_tokens)

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(1, 1, 1, -1).expand(b, h, num_noise_latents, -1)
    )#[:, :, :, 0]  # (b, num_heads, num_noise_latents, max_num_objects)
    # object_token_attn_prob = object_token_attn_prob / (torch.sum(object_token_attn_prob, dim=-1, keepdim=True) + 1e-6)
    object_token_attn_prob = object_token_attn_prob.squeeze(-1)[1:]

    # [BalancedL1Loss] calculate background / object segmap
    motion_map = motion_map.unsqueeze(1).expand(b, h, num_noise_latents)[1:]
    background_segmaps = 1 - motion_map
    background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
    object_segmaps_sum = motion_map.sum(dim=2) + 1e-5

    # [BalancedL1Loss] get loss
    # background_loss = (object_token_attn_prob * background_segmaps).sum(dim=2) / background_segmaps_sum
    # object_loss = (object_token_attn_prob * motion_map).sum(dim=2) / object_segmaps_sum
    # loss = background_loss - object_loss  # (b, num_heads)
    # loss = loss.mean()
    loss = ((object_token_attn_prob - motion_map) ** 2).sum(-1).mean((0, 1))

    return loss


def get_object_localization_loss(controller, motion_map, image_token_idx, tag=''):
    cross_attention_scores = {
        key: torch.stack([item for item in controller.step_store[key]]).sum(dim=0) for key in controller.step_store
        if "cross" in key
    }
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        # motion_map = motion_maps[k.replace("cross", "frame")]
        # motion_map = motion_map.reshape(-1, int(motion_map.shape[-1] ** 0.5), int(motion_map.shape[-1] ** 0.5))
        layer_loss = get_object_localization_loss_for_one_layer(v, motion_map, image_token_idx, controller, tag + k)
        loss += layer_loss
    return loss / num_layers


@torch.no_grad()
def get_object_localization_motion(controller, video_length, heads, object_masks=None, unet_layer_names=None, threshold=2.5):
    if unet_layer_names is None:
        # use all layers in controller.step_store
        unet_layer_names = ['']

    frame_attention_keys = [
        key for key in controller.step_store if any([f"{layer}.frame" in key for layer in unet_layer_names])
    ]

    # Calculate motion map
    motion_maps = []
    # for name_and_type, score in controller.step_store.items():
    for name in frame_attention_keys:
        score = controller.step_store.pop(name)
        res = int(score.shape[-1] ** 0.5)
        out = score.reshape(video_length, heads, res ** 2, res ** 2)
        out = out.permute(1, 0, 2, 3)
        out = out.sum(0) / out.shape[0]
        row_max = torch.argmax(out, dim=-1) // res  # (frame, res, res)
        col_max = torch.argmax(out, dim=-1) % res
        row_ind = torch.arange(0, res).repeat_interleave(res)[None, :].to(score.device)
        col_ind = torch.arange(0, res).repeat(res)[None, :].to(score.device)
        dist = torch.sqrt((row_max - row_ind) ** 2 + (col_max - col_ind) ** 2)
        dist = F.interpolate(dist.reshape(-1, 1, res, res), size=(64, 64), mode="bilinear", antialias=True)
        motion_maps += [dist.squeeze(1)]  # squeeze channel-dimension
    motion_maps = torch.mean(torch.stack(motion_maps, 0), 0)
    motion_maps = (motion_maps > threshold).float()

    if object_masks is not None:
        masks = F.interpolate(object_masks, (64, 64)).squeeze(0)
        motion_maps *= masks

        # [Visualize]
        # import utils.ptp_utils as ptp_utils
        # import numpy as np
        # from PIL import Image
        # images = []
        # # size = int(dist.shape[-1] ** 0.5)
        # for i in range(video_length):  # F_i frame
        #     image = motion_maps[i]  # dist[i, :].reshape(size, size)
        #     image = 255 * image / image.max()
        #     image = image.unsqueeze(-1).expand(*image.shape, 3)
        #     image = image.detach().cpu().numpy().astype(np.uint8)
        #     image = np.array(Image.fromarray(image).resize((256, 256)))
        #     images.append(image)
        # ptp_utils.view_images(np.stack(images, axis=0), set_title=f'{controller.cur_step} training step')

    return motion_maps
