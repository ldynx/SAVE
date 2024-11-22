# Adapted from https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

import cv2
import re
import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F

from typing import Union, Tuple
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from torch.optim.adam import Adam
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from transformers import CLIPTextModel, CLIPTokenizer

from utils.tuneavideo.models.unet import UNet3DConditionModel
from utils.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from utils.ptp import (
    load_512_seq, AttentionStore, AttentionStore__, LocalBlend, register_attention_control_exp
)
from utils.time_embedding_utils import (
    switch_text_model_forward_with_time_embedding,
    switch_pipeline_encode_with_time_embedding,
)

scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
)
MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# need to adjust sometimes
mask_th = (.3, .3)


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def latent2image_video(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents[0].permute(1, 0, 2, 3)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=self.model.vae.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def image2latent_video(self, image):
        with torch.no_grad():
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(0, 3, 1, 2).to(device, dtype=self.model.vae.dtype)
            latents = self.model.vae.encode(image)['latent_dist'].mean
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)
            latents = latents * 0.18215
        return latents

    ######## TEXTUAL INVERSION ########
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        self.context = self.model._encode_prompt(
            prompt, self.model.device, num_videos_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None
        )
        self.prompt = prompt
    ###################################

    @torch.no_grad()
    def ddim_loop(self, latent):
        if isinstance(self.context, list):
            # use Prompt+
            cond_embeddings = [context.chunk(2)[0] for context in self.context]
        else:
            uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent_video(image)
        image_rec = self.latent2image_video(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10, early_stop_epsilon=1e-5,
               verbose=False):
        self.init_prompt(prompt)
        image_gt = load_512_seq(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    def invert_(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), verbose=False, n_sample_frame=8, start_frame=0):
        self.init_prompt(prompt)
        image_gt = load_512_seq(image_path, *offsets, n_sample_frame=n_sample_frame, start_frame=start_frame)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        return (image_gt, image_rec), ddim_latents[-1], None

    def __init__(self, model):
        self.model = model
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def main(
    pretrained_model_path: str,
    image_path: str,
    prompt: str,
    prompts: Tuple[str],
    save_name: str = '',
    blend_word: Tuple[str] = None,
    video_len: int = 8,
    fast: bool = False,
    mixed_precision: str = 'fp32',
    ######## TEXTUAL INVERSION ########
    placeholder_tokens: str = None,
    sentence_component: str = None,
    ###################################
):
    # Define Output folder
    output_folder = os.path.join(pretrained_model_path, 'results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert prompts from ListConfig to list
    prompts = list(prompts)

    # Mixed precision
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    ).to(device, dtype=weight_dtype)
    unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet"
    ).to(device)

    # Get placeholder tokens
    placeholder_tokens = placeholder_tokens.split("|")
    sentence_component = sentence_component.split("|")

    placeholder_token_ids = []
    object_indices = []
    motion_indices = []
    for token, component in zip(placeholder_tokens, sentence_component):
        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)
        placeholder_token_ids.append(placeholder_token_id)
        motion_indices += [True] if component == "<v>" else [False]
        object_indices += [True] if component == "<o>" else [False]
    placeholder_token_ids = torch.tensor(placeholder_token_ids)
    motion_updates = torch.isin(torch.arange(len(tokenizer)), placeholder_token_ids[motion_indices])

    # Initialize time parameters
    text_emb_dim = text_encoder.config.hidden_size
    time_proj = Timesteps(num_channels=text_emb_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedding = TimestepEmbedding(in_channels=text_emb_dim * 2, time_embed_dim=text_emb_dim)
    ckpt = torch.load(os.path.join(pretrained_model_path, "time_embedding.pt"), map_location='cpu')
    time_embedding.load_state_dict(ckpt)
    time_embedding.requires_grad_(False)
    time_embedding = time_embedding.to(device)

    # Get model pipeline
    ldm_stable = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
    ).to(device)

    # Update text_encoder forward and prompt encoding
    ldm_stable.text_encoder.text_model.forward = switch_text_model_forward_with_time_embedding(
        text_encoder.text_model, time_embedding
    )
    ldm_stable._encode_prompt = switch_pipeline_encode_with_time_embedding(
        ldm_stable, time_proj, motion_updates, video_len
    )

    # Null-text inversion
    null_inversion = NullInversion(ldm_stable)

    # [Out of Memory] split video frames
    start_frame = 0
    if fast:
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert_(
            image_path, prompt, offsets=(0, 0, 0, 0), verbose=True, n_sample_frame=video_len, start_frame=start_frame
        )
    else:
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(
            image_path, prompt, offsets=(0, 0, 0, 0), verbose=True
        )

    # Run inference !!
    generator = torch.Generator(device=device)
    sequences = []
    for i, edit_prompt in enumerate(prompts[1:]):
        if blend_word is not None:
            blend_word_per_prompt = (((blend_word[0],), (blend_word[i + 1],))) if blend_word else None
            lb = LocalBlend([prompt, edit_prompt], blend_word_per_prompt, tokenizer, th=mask_th)
            controller = AttentionStore__(local_blend=lb)
            register_attention_control_exp(ldm_stable, controller)
        else:
            controller = AttentionStore()
            register_attention_control_exp(ldm_stable, controller)

        with torch.no_grad():
            sequence = ldm_stable(
                [prompt, edit_prompt],
                generator=generator,
                latents=x_t,
                controller=controller,
                uncond_embeddings_pre=uncond_embeddings,
                video_length=video_len,
                fast=fast,
            ).videos

        if i == 0:
            sequence_ = rearrange(sequence[0], "c t h w -> t h w c")
            sequence_list = []
            for j in range(sequence_.shape[0]):
                sequence_list.append(Image.fromarray((sequence_[j] * 255).numpy().astype(np.uint8)))

            sequence_list[0].save(
                os.path.join(output_folder, f"{prompts[0].replace(' ', '_')}{save_name}.gif"),
                save_all=True, append_images=sequence_list[1:], optimize=False, loop=0, duration=250
            )

        sequence = rearrange(sequence[1], "c t h w -> t h w c")
        sequence_list = []
        for j in range(sequence.shape[0]):
            sequence_list.append(Image.fromarray((sequence[j] * 255).numpy().astype(np.uint8)))
        sequences += [sequence_list]
        # Save output
        sequence_list[0].save(
            os.path.join(output_folder, f"{prompts[i + 1].replace(' ', '_')}{save_name}.gif"),
            save_all=True, append_images=sequence_list[1:], optimize=False, loop=0, duration=250
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/utils.yaml")
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--save_name", type=str, default='')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config), fast=args.fast, save_name=args.save_name)
