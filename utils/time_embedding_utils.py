import re
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling

from utils.tuneavideo.models.unet import UNet3DConditionModel
from utils.ti_utils import _expand_mask, _make_causal_mask


def register_placeholder_token(tokenizer, text_encoder, token, init_token):
    # Add tokens to tokenizer and text_encoder
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    # Get token id
    placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

    # Initialize tokens
    if init_token.startswith("<rand"):
        # <rand-"sigma">, e.g. <rand-0.5>
        sigma_val = float(re.findall(r"<rand-(.*)>", init_token)[0])
        token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
        )
        print(
            f"Initialized {token} with random noise (sigma={sigma_val}), empirically "
            f"{token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
        )
        print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")
    elif init_token == "<zero>":
        token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
    elif init_token.isnumeric():
        token_embeds[placeholder_token_id] = token_embeds[int(init_token)]
    else:
        token_ids = tokenizer.encode(init_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            # raise ValueError("The initializer token must be a single token.")
            token_ids = [token_ids[-1]]
        initializer_token_id = token_ids[0]
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    return tokenizer, text_encoder, placeholder_token_id


def expand_2d_to_3d(unet):
    config = dict(unet.config)
    config["_class_name"] = "UNet3DConditionModel"
    config["down_block_types"] = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D"
    ]
    config["up_block_types"] = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D"
    ]
    if "mid_block_type" in config:
        config.pop("mid_block_type")

    model = UNet3DConditionModel.from_config(config)
    state_dict = unet.state_dict()
    model.set_attn_processor(unet.attn_processors)
    for k, v in model.state_dict().items():
        if '_temp.' in k:
            state_dict.update({k: v})
    model.load_state_dict(state_dict)

    return model


def encode_with_time_embedding(placeholder_embedding, time_embedding, positional_encoding, lambda_=0):
    # output_embedding = time_embedding(placeholder_embedding + positional_encoding)
    if placeholder_embedding.shape[0] != positional_encoding.shape[0]:
        placeholder_embedding = placeholder_embedding.expand(positional_encoding.shape[0], -1)
    output_embedding = time_embedding(torch.cat([placeholder_embedding, positional_encoding], -1))

    # Normalize embedding
    pre_norm = output_embedding.norm(dim=-1, keepdim=True)
    output_embedding = F.normalize(output_embedding) * (pre_norm + lambda_ * (0.4 - pre_norm))
    return output_embedding


def switch_text_model_forward_with_time_embedding(self, time_embedding, lr_scheduler=None):
    """
    Switch forward function for <text_encoder.text_model>.
    Attain `token_embedding[placeholder_idx]` from `positional_encoding` and `time_embedding`.
    """
    def forward(
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        positional_encoding: Optional[torch.Tensor] = None,
        placeholder_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        ######## TEXTUAL INVERSION ########
        if positional_encoding is not None and placeholder_idx is not None:
            batch_size, seq_length = input_ids.shape
            video_len = positional_encoding.shape[0]

            if position_ids is None:
                position_ids = self.embeddings.position_ids[:, :seq_length]

            orig_inputs_embeds = self.embeddings.token_embedding(input_ids)
            orig_inputs_embeds = orig_inputs_embeds.unsqueeze(1).expand(-1, video_len, -1, -1)
            inputs_embeds = orig_inputs_embeds.clone()
            for b in range(len(placeholder_idx[0])):
                placeholder_embedding = inputs_embeds[placeholder_idx[0][b], :, placeholder_idx[1][b], :]
                placeholder_embedding = encode_with_time_embedding(
                    placeholder_embedding, time_embedding, positional_encoding,
                    lambda_=min(1.0, 100 * lr_scheduler.get_last_lr()[0]) if lr_scheduler else 0
                )
                inputs_embeds[placeholder_idx[0][b], :, placeholder_idx[1][b], :] = placeholder_embedding

            position_embeddings = self.embeddings.position_embedding(position_ids)
            hidden_states = inputs_embeds + position_embeddings.unsqueeze(1)

            # hidden_states = hidden_states.expand(batch_size * video_len, -1, -1)
            hidden_states = hidden_states.view(batch_size * video_len, seq_length, -1)
            input_shape = input_ids.repeat(video_len, 1).size()

            if torch.sum(torch.isnan(hidden_states)) > 0:
                print()
        ###################################
        else:
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        ######## TEXTUAL INVERSION ########
        if positional_encoding is not None and placeholder_idx is not None:
            # Reshape and split batch and frame dimension
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).repeat(video_len, 1).argmax(dim=-1),
            ]
            pooled_output = pooled_output.view(batch_size, video_len, -1)
            last_hidden_state = last_hidden_state.view(batch_size, video_len, seq_length, -1)
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        ###################################

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    return forward


def switch_pipeline_encode_with_time_embedding(self, time_proj, index_updates, video_len):
    """
    Attain prompt embedding with `token_embedding[placeholder_idx]` from `positional_encoding` and `time_embedding`.
    Use <text_encoder.text_model> switched `forward` as `forward_time_embedding`.
    """
    def _encode_prompt(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        # Get token ids
        prompt_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Use time embeddings
        positional_encoding = time_proj(torch.arange(0, video_len, device=device))
        placeholder_idx = torch.where(
            torch.isin(prompt_ids, torch.arange(len(self.tokenizer))[index_updates].to(device))
        )
        text_embeddings = self.text_encoder.text_model(
            prompt_ids, positional_encoding=positional_encoding, placeholder_idx=placeholder_idx
        )[0]

        # Reshape time embeddings considering num_videos_per_prompt
        bs_embed, _, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.unsqueeze(2).repeat(1, 1, num_videos_per_prompt, 1, 1).permute(0, 2, 1, 3, 4)
        text_embeddings = text_embeddings.reshape(bs_embed * num_videos_per_prompt, video_len, seq_len, -1)

        # Classifier free guidance
        if do_classifier_free_guidance:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids
            uncond_embeddings = self.text_encoder.text_model.embeddings.token_embedding(uncond_input_ids.cuda())

            # Reshape uncond_embeddings with video dimension
            uncond_embeddings = uncond_embeddings.unsqueeze(1)
            uncond_embeddings = uncond_embeddings.unsqueeze(2).repeat(1, video_len, num_videos_per_prompt, 1, 1)
            uncond_embeddings = uncond_embeddings.permute(0, 2, 1, 3, 4)
            uncond_embeddings = uncond_embeddings.reshape(bs_embed * num_videos_per_prompt, video_len, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    return _encode_prompt


def switch_text_model_forward_with_multi_time_embedding(self, time_embedding1, time_embedding2, lr_scheduler=None):
    """
    Switch forward function for <text_encoder.text_model>.
    Attain `token_embedding[placeholder_idx]` from `positional_encoding` and `time_embedding`.
    """
    def forward(
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        positional_encoding: Optional[torch.Tensor] = None,
        placeholder_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        ######## TEXTUAL INVERSION ########
        if positional_encoding is not None and placeholder_idx is not None:
            batch_size, seq_length = input_ids.shape
            video_len = positional_encoding.shape[0]

            if position_ids is None:
                position_ids = self.embeddings.position_ids[:, :seq_length]

            orig_inputs_embeds = self.embeddings.token_embedding(input_ids)
            orig_inputs_embeds = orig_inputs_embeds.unsqueeze(1).expand(-1, video_len, -1, -1)
            inputs_embeds = orig_inputs_embeds.clone()
            for b in range(len(placeholder_idx[0]) // 2):
                # Encode the first motion
                placeholder_embedding1 = inputs_embeds[placeholder_idx[0][2 * b], :, placeholder_idx[1][2 * b], :]
                placeholder_embedding1 = encode_with_time_embedding(
                    placeholder_embedding1, time_embedding1, positional_encoding,
                    lambda_=min(1.0, 100 * lr_scheduler.get_last_lr()[0]) if lr_scheduler else 0
                )
                inputs_embeds[placeholder_idx[0][0], :, placeholder_idx[1][0], :] = placeholder_embedding1
                # Encode the second motion
                placeholder_embedding2 = inputs_embeds[placeholder_idx[0][2 * b + 1], :, placeholder_idx[1][2 * b + 1], :]
                placeholder_embedding2 = encode_with_time_embedding(
                    placeholder_embedding2, time_embedding2, positional_encoding,
                    lambda_=min(1.0, 100 * lr_scheduler.get_last_lr()[0]) if lr_scheduler else 0
                )
                inputs_embeds[placeholder_idx[0][1], :, placeholder_idx[1][1], :] = placeholder_embedding2

            position_embeddings = self.embeddings.position_embedding(position_ids)
            hidden_states = inputs_embeds + position_embeddings.unsqueeze(1)

            # hidden_states = hidden_states.expand(batch_size * video_len, -1, -1)
            hidden_states = hidden_states.view(batch_size * video_len, seq_length, -1)
            input_shape = input_ids.repeat(video_len, 1).size()

            if torch.sum(torch.isnan(hidden_states)) > 0:
                print()
        ###################################
        else:
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        ######## TEXTUAL INVERSION ########
        if positional_encoding is not None and placeholder_idx is not None:
            # Reshape and split batch and frame dimension
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).repeat(video_len, 1).argmax(dim=-1),
            ]
            pooled_output = pooled_output.view(batch_size, video_len, -1)
            last_hidden_state = last_hidden_state.view(batch_size, video_len, seq_length, -1)
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        ###################################

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    return forward
