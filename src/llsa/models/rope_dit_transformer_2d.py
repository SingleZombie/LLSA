# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass


import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging, BaseOutput
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, Timesteps, TimestepEmbedding, LabelEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, FP32LayerNorm

from ..kernel.torch_op.flash_sparse_attention_res_1 import flash_sparse_residual_attention_l1_op
from ..kernel.torch_op.flash_sparse_attention_res_2 import flash_sparse_residual_attention_l2_op
from ..kernel.triton.rope import my_rope_fn
from ..kernel.triton.mean_pool import mean_pool1d

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

THETA = 10000


@dataclass
class MyTransformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or 
            `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821
    repa_features: "List[torch.Tensor]"  # noqa: F821


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def gen_rope_freq(dim: int, theta: int):
    scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    omega = 1.0 / (theta**scale)
    return omega


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    omega = gen_rope_freq(dim, theta).to(pos.device)

    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([sin_out, cos_out], dim=-1)
    stacked_out = stacked_out.reshape(-1, dim)

    return stacked_out


class RopeEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]

        if len(ids.shape) == 3:
            ids = ids[0]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(n_axes)],
            dim=-1,
        )
        return emb


class RopeAttnProcessor:

    PIXEL = 1
    SPARSE_L1 = 2
    SPARSE_L2 = 3

    ATTN_ENUM_MAP = {
        'pixel': PIXEL,
        'sparse_l1': SPARSE_L1,
        'sparse_l2': SPARSE_L2
    }

    def __init__(self, max_size: Tuple[int, int], rope_func,
                 attn_type: str = 'pixel',
                 patch_to_perms=None,
                 patch_to_bwd_perms=None,
                 sparse_topk=16,
                 sparse_block_size=16):
        self.max_size = max_size
        map_dict = RopeAttnProcessor.ATTN_ENUM_MAP
        try:
            self.attn_type = map_dict[attn_type]
        except KeyError:
            raise KeyError(
                f'Please select correct key from {list(map_dict.keys())} '
                'for RopeAttnProcessor')
        self.patch_size = 1

        self.rope_func = rope_func
        self.patch_to_rope_emb = {}

        if isinstance(sparse_topk, int):
            sparse_topk = [sparse_topk] * 3

        self.patch_to_perms = patch_to_perms
        self.patch_to_bwd_perms = patch_to_bwd_perms
        self.sparse_topk = sparse_topk
        self.sparse_block_size = sparse_block_size

        # self.res_attn_func = FlashSparseResidualAttentionL2(4)

    def register_rope_emb(self, patch_size, device, dtype):
        # TODO: 2d ID
        id_1d = torch.arange(
            self.max_size[0], dtype=torch.float32)
        new_size = int(self.max_size[0] / patch_size)
        id_1d = F.interpolate(
            id_1d[None, None, :], new_size, mode='linear').squeeze()
        image_ids = torch.zeros(new_size,
                                new_size, 2)
        image_ids[..., 0] = image_ids[..., 0] + id_1d[:, None]
        image_ids[..., 1] = image_ids[..., 1] + id_1d[None, :]

        image_ids = image_ids.reshape(
            1, int((self.max_size[0] / patch_size)**2), -1)

        if self.patch_to_perms is not None:
            perms = self.patch_to_perms[new_size]
            image_ids = image_ids[:, perms, :]

        self.patch_to_rope_emb[patch_size] = self.rope_func(
            image_ids).to(device).to(dtype)

    def fetch_rope_emb(self, patch_size, device, dtype=torch.float64):
        if patch_size not in self.patch_to_rope_emb:
            self.register_rope_emb(patch_size, device, dtype)
        return self.patch_to_rope_emb[patch_size]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim

        assert input_ndim == 3
        batch_size, seq_len, channel = hidden_states.shape
        height = int(seq_len ** 0.5)

        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(
        #         batch_size, channel, height * width).transpose(1, 2)
        # elif input_ndim == 3:
        #     batch_size, seq_len, channel = hidden_states.shape
        #     height = width = int(seq_len ** 0.5)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        pixel_size = self.max_size[0] / height

        q_inner_dim = query.shape[-1]
        q_head_dim = q_inner_dim // attn.heads

        v_inner_dim = value.shape[-1]
        v_head_dim = v_inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           q_head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, q_head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           v_head_dim).transpose(1, 2)

        input_dtype = query.dtype
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(input_dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(input_dtype)

        blsz = self.sparse_block_size

        if self.attn_type == RopeAttnProcessor.PIXEL:

            image_rotary_emb = self.fetch_rope_emb(
                pixel_size, query.device, query.dtype)
            query = my_rope_fn(query, image_rotary_emb)
            key = my_rope_fn(key, image_rotary_emb)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False)

        elif self.attn_type == RopeAttnProcessor.SPARSE_L1:

            image_rotary_emb_0 = self.fetch_rope_emb(
                1 * pixel_size, query.device, query.dtype)

            query = my_rope_fn(query, image_rotary_emb_0)
            key = my_rope_fn(key, image_rotary_emb_0)

            pq = mean_pool1d(query, blsz)
            pk = mean_pool1d(key, blsz)
            pv = mean_pool1d(value, blsz)

            hidden_states = flash_sparse_residual_attention_l1_op(query, key, value, pq, pk, pv,
                                                                  self.sparse_topk[1], blsz, blsz)

        elif self.attn_type == RopeAttnProcessor.SPARSE_L2:
            image_rotary_emb_0 = self.fetch_rope_emb(
                1 * pixel_size, query.device, query.dtype)

            query = my_rope_fn(query, image_rotary_emb_0)
            key = my_rope_fn(key, image_rotary_emb_0)

            pq1 = mean_pool1d(query, blsz)
            pk1 = mean_pool1d(key, blsz)
            pv1 = mean_pool1d(value, blsz)

            pq2 = mean_pool1d(pq1, blsz)
            pk2 = mean_pool1d(pk1, blsz)
            pv2 = mean_pool1d(pv1, blsz)

            hidden_states = flash_sparse_residual_attention_l2_op(query, key, value, pq1, pk1, pv1, pq2, pk2, pv2,
                                                                  self.sparse_topk[1],
                                                                  self.sparse_topk[0], blsz, blsz * blsz, blsz)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * v_head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, embedding_dim, num_classes=None, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim)
        if num_classes is not None:
            self.class_embedder = LabelEmbedding(
                num_classes, embedding_dim, class_dropout_prob)
            self.use_class_cond = True
        else:
            self.use_class_cond = False

    def forward(self, timestep, class_labels=None, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb

        if self.use_class_cond:
            if class_labels is None:
                raise ValueError(
                    'Claa labels cannot be `None` for class conditioned embedding')
            class_labels = self.class_embedder(class_labels)  # (N, D)
            conditioning = conditioning + class_labels  # (N, D)

        return conditioning


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(
            embedding_dim, num_embeddings)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(
                embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels,
                           hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}."
            f"Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice)
            for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


@maybe_allow_in_graph
class RopeTransformerBlock(nn.Module):
    r"""
    A RoPE Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        rope_processor,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(
                dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            processor=rope_processor,
            qk_norm=qk_norm
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(
                    dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            if norm_type == "ada_norm_single":  # For Latte
                self.norm2 = nn.LayerNorm(
                    dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(
                torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # self.cached_x = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] +
                timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * \
                (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy(
        ) if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * \
                (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * \
                (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


def gen_permuatations(log_num_tokens: int):
    # num_tokens = 4 ** (1 + log_num_tokens)
    perm = torch.tensor([[0, 1], [2, 3]])
    base_num = 4

    for i in range(log_num_tokens):
        length = perm.shape[-1]
        perm = perm[None, :, :].expand(
            4, -1, -1) + torch.arange(0, 4)[:, None, None] * base_num
        perm = perm.reshape(2, 2, length, length)
        perm = rearrange(perm, 'a b c d -> a c b d')
        perm = perm.reshape(length * 2, length * 2)

        base_num *= 4

    perm = perm.flatten()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    return perm, inv_perm


class RopeDiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        patch_size: int = 1,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        axes_dims_rope=[36, 36],
        attn_type='pixel',
        token_permutation: bool = False,  # reordering
        sparse_topk=16,
        rope_size: Optional[int] = None,
        enable_repa: bool = False,
        repa_dim: int = 768,
        repa_layer_id: int = 5,
        repa_size: Optional[int] = None,
        qk_norm: Optional[str] = None,
        sparse_block_size=16,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size
        self.sample_size = sample_size

        self.patch_size = patch_size

        self.pos_embed = RopeEmbedding(
            dim=self.inner_dim, theta=THETA, axes_dim=axes_dims_rope)
        self.proj_in = nn.Conv2d(in_channels, self.inner_dim, kernel_size=(
            patch_size, patch_size), stride=patch_size)

        if rope_size is None:
            rope_size = sample_size // patch_size

        self.token_permutation = token_permutation

        if self.token_permutation:
            patch_to_perms = {}
            patch_to_bwd_perms = {}
            self.fwd_perms = {}
            self.bwd_perms = {}

            for log2_scale in range(2, 10):

                img_size = 2 ** log2_scale
                log_scale_m1 = log2_scale - 1
                perm, inv_perm = gen_permuatations(int(log_scale_m1))
                self.fwd_perms[img_size] = inv_perm
                self.bwd_perms[img_size] = perm

                patch_to_perms[img_size] = inv_perm
                patch_to_bwd_perms[img_size] = perm
        else:
            patch_to_perms = None
            patch_to_bwd_perms = None
            self.fwd_perms = None
            self.bwd_perms = None

        processor = RopeAttnProcessor(
            (rope_size, rope_size),
            self.pos_embed,
            attn_type,
            patch_to_perms,
            patch_to_bwd_perms,
            sparse_topk,
            sparse_block_size)

        self.transformer_blocks = nn.ModuleList(
            [
                RopeTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    processor,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        if enable_repa:
            self.repa_layer = build_mlp(self.inner_dim, 2048, repa_dim)
            self.repa_layer_id = repa_layer_id

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.patch_size *
            self.patch_size * self.out_channels
        )

        self.enable_repa = enable_repa
        self.repa_size = repa_size

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
        return_repa_feature=False
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of
                shape `(batch size, channel, height, width)` if continuous): Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_size, _, height, width = hidden_states.shape
        patch_height = height // self.patch_size

        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, -1, height * width // (self.patch_size)**2)
        hidden_states = hidden_states.transpose(1, 2)

        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timestep as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = hidden_states.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timestep = torch.tensor(
                [timestep], dtype=dtype, device=hidden_states.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(hidden_states.shape[0])

        if self.token_permutation:
            fwd_perm = self.fwd_perms[patch_height].to(hidden_states.device)
            hidden_states = hidden_states[:, fwd_perm, :]

        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                def inference(x):
                    return block(
                        x,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

            if return_repa_feature and self.enable_repa and i == self.repa_layer_id:
                x = hidden_states
                if self.repa_size is not None:
                    if self.token_permutation:
                        ds_scale = patch_height // self.repa_size
                        bwd_perm = self.bwd_perms[self.repa_size]
                        x = mean_pool1d(x, ds_scale**2, 'bsc')[:, bwd_perm]
                    else:
                        raise NotImplementedError(
                            'Please set self.token_permuation=true for efficent REPA pooling')

                repa_feature = self.repa_layer(x)

        if self.token_permutation:
            bwd_perm = self.bwd_perms[patch_height]
            hidden_states = hidden_states[:, bwd_perm, :]

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(
            hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size,
                   self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height *
                   self.patch_size, width * self.patch_size)
        )

        if return_repa_feature and self.enable_repa:
            return (output, repa_feature)

        else:
            if not return_dict:
                return (output,)

            return Transformer2DModelOutput(sample=output)
