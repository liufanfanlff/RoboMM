import torch.nn as nn
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import *

class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False, residual=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self.residual = residual

        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def clone_parameters(self):
        self.res_layer = copy.deepcopy(self.gated_cross_attn_layer)
        if self.res_layer is not None:
            self.res_layer.requires_grad_(False)

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )
            
            # Residual
            if self.residual and self.res_layer is not None:
                lang_x_res = self.res_layer(
                    lang_x,
                    self.vis_x,
                    media_locations=self.media_locations,
                    attend_previous=self.attend_previous,
                )
                lang_x = (lang_x + lang_x_res) / 2.0

        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """
    
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        residual=False,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        print('-'*100)
        print(self.decoder_layers_attr_name)
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, dim_visual=vis_hidden_size
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(gradient_checkpointing, residual=residual)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing, residual=False):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing, residual=residual
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        media_locations = input_ids == self.media_token_id

        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask # 所有的为True
        # return super().forward(**kwargs)  # Call the other parent's forward method
        return self.forward_super(**kwargs)

    ############################################
    def forward_super(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MosaicGPT
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MosaicGPT')
        if output_attentions:
            raise NotImplementedError(
                'output_attentions is not implemented yet for MosaicGPT')

        if attention_mask is not None and attention_mask[:, 0].sum(
        ) != attention_mask.shape[0] and self.training:
            raise NotImplementedError(
                'MosaicGPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                'prefix_mask is a required argument when MosaicGPT is configured with prefix_lm=True.'
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MosaicGPT is configured with attn_uses_sequence_id=True ' +\
                    'and the model is in train mode.'
                )
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'MosaicGPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' +\
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.'
                )

        S = input_ids.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb # 默认没有添加pos编码
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention ' +\
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )
                # get the key tensor whose spec should be (batch, seq, dim), and
                # collect the `seq`, so that the position embedding is shifted
                past_position = past_key_values[0][0].size(1)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )
            pos = torch.arange(past_position,
                               S + past_position,
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(pos - torch.cumsum(
                    (~attention_mask).to(torch.int32), dim=1)[:,
                                                              past_position:],
                                  min=0)

            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore 使用drop mask掉部分输入，但drop rate目前为0
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=x.dtype,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id)

        # initialize the past key values cache if it should be used
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        for b_idx, block in enumerate(self.transformer.blocks):  # type: ignore
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,) # x表示语言
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None
            x, past_key_value = block(x,
                                      past_key_value=past_key_value,
                                      attn_bias=attn_bias,
                                      attention_mask=attention_mask,
                                      is_causal=self.is_causal)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x)  # type: ignore

        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        return CausalLMOutputWithPast(logits=logits,
                                      past_key_values=past_key_values,
                                      hidden_states=all_hidden_states)

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clone_parameters(self):
        for layer in self._get_decoder_layers():
            layer.clone_parameters()

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
