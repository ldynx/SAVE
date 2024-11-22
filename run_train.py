# From https://github.com/showlab/Tune-A-Video/blob/main/train_tuneavideo.py

import argparse
import logging
import inspect
import os
import math
import itertools
import diffusers
import transformers
import torchvision
import torch.utils.checkpoint
import torch.nn.functional as F

from omegaconf import OmegaConf
from typing import Optional, Tuple, Dict
from einops import rearrange
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.pipelines import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from utils.tuneavideo.models.unet import UNet2DConditionModel
from utils.tuneavideo.data.step_training_dataset import StepTrainingSepDataset
from utils.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from utils.time_embedding_utils import (
    register_placeholder_token,
    switch_text_model_forward_with_time_embedding,
    expand_2d_to_3d
)
# exp_use_attention_localization
from utils.localization_utils import (
    LayerAttentionStore,
    register_layer_attention_store,
    get_object_localization_motion,
    get_object_localization_loss,
)


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    s1_learning_rate: float = 5e-4,
    learning_rate: float = 2e-6,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    ######## TEXTUAL INVERSION ########
    placeholder_tokens: str = None,
    initializer_tokens: str = None,
    sentence_component: str = None,
    num_s1_train_epochs: int = 250,
    exp_use_attention_localization: bool = False,
    exp_localization_weight: float = 1e-4,
    ###################################
    exp_name: str = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard", project_dir=os.path.join(output_dir, exp_name)
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if exp_name:
        output_dir = os.path.join(output_dir, exp_name)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # Get placeholder tokens
    placeholder_tokens = placeholder_tokens.split("|")
    initializer_tokens = initializer_tokens.split("|")
    sentence_component = sentence_component.split("|")

    # Initialize and register tokens
    placeholder_token_ids = []
    placeholder_token_keys = []
    motion_indices = []
    object_indices = []
    protagonists, protagonist_names = [], []
    for token, init_token, component in zip(placeholder_tokens, initializer_tokens, sentence_component):
        tokenizer, text_encoder, token_id = register_placeholder_token(tokenizer, text_encoder, token, init_token)
        placeholder_token_ids += [token_id]
        placeholder_token_keys += [token]
        motion_indices += [True] if component == "<v>" else [False]
        object_indices += [True] if component == "<o>" else [False]
        if component == "<o>":
            protagonists += [init_token]
            protagonist_names += [token + ' ' + init_token]
    placeholder_token_ids = torch.tensor(placeholder_token_ids)
    motion_updates = torch.isin(torch.arange(len(tokenizer)), placeholder_token_ids[motion_indices])
    object_updates = torch.isin(torch.arange(len(tokenizer)), placeholder_token_ids[object_indices])

    # Define update text embeddings
    index_updates = object_updates
    index_no_updates = ~index_updates

    # Initialize time parameters
    text_emb_dim = text_encoder.config.hidden_size
    time_proj = Timesteps(num_channels=text_emb_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    positional_encoding = time_proj(torch.arange(0, train_data.n_sample_frames))
    time_embedding = TimestepEmbedding(in_channels=text_emb_dim * 2, time_embed_dim=text_emb_dim)

    # Freeze vae
    vae.requires_grad_(False)

    # Unfreeze **2D** unet
    unet.requires_grad_(False)

    # Unfreeze text encoder embedding
    text_encoder.requires_grad_(True)
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # Unfreeze time embedding
    time_embedding.requires_grad_(True)

    # Training settings
    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Initialize the optimizer
    params_to_optimize = [
        {'params': unet.parameters()}, {'params': text_encoder.get_input_embeddings().parameters()}
    ]
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=s1_learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = StepTrainingSepDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
        return_tensors="pt"
    ).input_ids
    train_dataset.s1_prompt_ids = tokenizer(
        f"a photo of {' and '.join(protagonist_names)}",
        max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # Set initializer and indices in dataset
    train_dataset.protagonists = protagonists

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
    )
    # Get the validation pipeline
    validation_pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path, ae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,  # TODO 0(p2p) vs. 100(ti)
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    positional_encoding = positional_encoding.to(accelerator.device, dtype=weight_dtype)

    # Save original text embeddings
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0

        ########### SECOND STEP ###########
        if epoch == num_s1_train_epochs:
            # Unfreeze **3D** unet
            unet = expand_2d_to_3d(unet)
            unet.requires_grad_(False)
            for name, module in unet.named_modules():
                if name.endswith(tuple(trainable_modules)):
                    for params in module.parameters():
                        params.requires_grad = True

            if enable_xformers_memory_efficient_attention:
                if is_xformers_available():
                    unet.enable_xformers_memory_efficient_attention()
                else:
                    raise ValueError("xformers is not available. Make sure it is installed correctly")

            # Initialize the optimizer
            params_to_optimize = [
                {'params': unet.parameters()}, {'params': time_embedding.parameters()},
                {'params': text_encoder.get_input_embeddings().parameters()}
            ]
            optimizer = optimizer_cls(
                params_to_optimize,
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon,
            )

            # Prepare everything with our `accelerator`.
            unet, optimizer, train_dataloader, lr_scheduler, time_embedding = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler, time_embedding
            )

            # Update text_encoder forward and train_dataset state
            text_encoder.text_model.forward = switch_text_model_forward_with_time_embedding(
                text_encoder.text_model, time_embedding
            )
            train_dataset.s1_training = False

            # Include motion token to update index
            index_updates = motion_updates
            index_no_updates = ~index_updates

        # Set controller and object masks for localization loss
        if exp_use_attention_localization and epoch == num_s1_train_epochs + 1:
            UNET_LAYER_NAMES = [
                "down_blocks.0.cross",
                "down_blocks.1.cross",
                "down_blocks.2.cross",
                "mid_block.cross",
                "up_blocks.1.cross",
                "up_blocks.2.cross",
                "up_blocks.3.cross",
                "up_blocks.2.frame",
                # "up_blocks.3.frame"
            ]
            controller = LayerAttentionStore(UNET_LAYER_NAMES)
            register_layer_attention_store(unet, controller, UNET_LAYER_NAMES)
        ###################################

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                if epoch >= num_s1_train_epochs:
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Get the text embedding for conditioning
                if epoch >= num_s1_train_epochs:
                    placeholder_idx = torch.where(torch.isin(
                        batch["prompt_ids"][0], torch.arange(len(tokenizer))[motion_updates].to(latents.device)
                    ))
                    encoder_hidden_states = text_encoder.text_model(
                        batch["prompt_ids"], positional_encoding=positional_encoding, placeholder_idx=placeholder_idx
                    )[0]
                else:
                    encoder_hidden_states = text_encoder.text_model(batch["prompt_ids"])[0].repeat(video_length, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Instance masked loss in [Break-A-Scene] for <s1> training
                if epoch < num_s1_train_epochs and batch["instance_masks"].shape[-1] > 0:
                    downsampled_mask = F.interpolate(input=batch["instance_masks"].float(), size=(64, 64))[0]
                    model_pred = model_pred * downsampled_mask.unsqueeze(1)  # (n_frame, n_channel, 64, 64)
                    target = target * downsampled_mask.unsqueeze(1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if exp_use_attention_localization and epoch > num_s1_train_epochs:
                    # Get cross-attention localization loss
                    head_dim = unet.config.attention_head_dim
                    motion_map = get_object_localization_motion(
                        controller, video_length, head_dim, batch["instance_masks"], ["up_blocks.2", "up_blocks.3"]
                    )
                    loss += exp_localization_weight * get_object_localization_loss(
                        controller, motion_map, placeholder_idx[1]
                    )

                    # Freeze controller while backward process
                    controller.freeze_while_backprop()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Normalize and update embeddings
                with torch.no_grad():
                    pre_norm = text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1, keepdim=True)
                    lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                    text_encoder.get_input_embeddings().weight[index_updates] = F.normalize(
                        text_encoder.get_input_embeddings().weight[index_updates, :], dim=-1,
                    ) * (pre_norm + lambda_ * (0.4 - pre_norm))
                    text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

                if exp_use_attention_localization and epoch > num_s1_train_epochs:
                    controller.delete_after_backprop()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step == num_s1_train_epochs - 1:
                    if accelerator.is_main_process:
                        validation_pipeline = validation_pipeline.to(accelerator.device)

                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)
                        prompts = [
                            f"a photo of {' and '.join(protagonist_names)}",
                            train_data.prompt.replace(''.join(
                                [placeholder_tokens[i] for i in range(len(motion_indices)) if motion_indices[i]]
                            ), initializer_tokens[sentence_component.index('<v>')])
                        ]
                        for idx, prompt in enumerate(prompts):
                            sample = validation_pipeline(
                                prompt,
                                generator=generator,
                                height=train_data.get("height", 512),
                                width=train_data.get("width", 512),
                                num_inference_steps=validation_data.get("num_inference_steps", 25),
                                guidance_scale=validation_data.get("guidance_scale", 8.),
                            ).images[0]
                            sample = torchvision.transforms.functional.to_tensor(sample)
                            samples.append(sample)

                        samples = torch.stack(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.png"
                        torchvision.utils.save_image(samples, save_path, nrow=4)
                        logging.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

        # Save time embedding
        torch.save(time_embedding.state_dict(), os.path.join(output_dir, "time_embedding.pt"))

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/cat-flower-train.yaml")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    main(
        **OmegaConf.load(args.config),
        exp_name=args.exp_name,
    )
