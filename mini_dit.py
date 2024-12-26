import jax, math, os, optax
from flax import nnx
from jax import Array, random as jrand, numpy as jnp
from einops import rearrange
from typing import List
from tqdm.auto import tqdm


class config:
    batch_size = 128
    img_size = 32
    seed = 33
    patch_size = (2, 2)
    lr = 1e-4
    epochs = 20
    data_split = 20_000
    cfg_scale = 2.0
    mini_data_id = "uoft-cs/cifar10"


JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20

rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
linear_init = nnx.initializers.xavier_uniform()
linear_bias_init = nnx.initializers.constant(1)


# modulation with shift and scale
def modulate(x_array: Array, shift, scale) -> Array:
    x = x_array * (1 + jnp.expand_dims(scale, 1))
    x = x + jnp.expand_dims(shift, 1)

    return x


# equivalnet of F.linear
def linear(array: Array, weight: Array, bias: Array | None = None) -> Array:
    out = jnp.dot(array, weight)

    if bias is not None:
        out += bias

    return out


# Adapted from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate(
            [jnp.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb


# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs=rngs,
        patch_size: int = 4,
        in_chan: int = 3,
        embed_dim: int = 1024,
        img_size=config.img_size,
        # rngs: nnx.Rngs=rngs
    ):
        super().__init__()
        self.patch_size = patch_size
        self.gridsize = tuple(
            [s // p for s, p in zip((img_size, img_size), (patch_size, patch_size))]
        )
        self.num_patches = self.gridsize[0] * self.gridsize[1]

        self.conv_project = nnx.Conv(
            in_chan,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=False,  # Typically, PatchEmbed doesn't use bias
            rngs=rngs,
        )

    def __call__(self, img: jnp.ndarray) -> jnp.ndarray:
        # Project image patches with the convolution layer
        x = self.conv_project(
            img
        )  # Shape: (batch_size, embed_dim, height // patch_size, width // patch_size)

        # Reshape to (batch_size, num_patches, embed_dim)
        batch_size, h, w, embed_dim = x.shape
        x = x.reshape(batch_size, -1)  # Shape: (batch_size, num_patches, embed_dim)

        return x


# embeds a flat vector
class VectorEmbedder(nnx.Module):
    def __init__(self, input_dim, hidden_size, rngs=rngs):
        super().__init__()
        self.linear_1 = nnx.Linear(input_dim, hidden_size, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, x: Array):
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.lin_1 = nnx.Linear(freq_embed_size, hidden_size, rngs=rngs)
        self.lin_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(time_array: Array, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half) / half)

        args = jnp.float_(time_array[:, None]) * freqs[None]

        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )

        return embedding

    def __call__(self, x: Array) -> Array:
        t_freq = self.timestep_embedding(x, self.freq_embed_size)
        t_embed = nnx.silu(self.lin_1(t_freq))

        return self.lin_2(t_embed)


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes, hidden_size, drop):
        super().__init__()
        use_cfg_embeddings = drop > 0
        self.embedding_table = nnx.Embed(
            num_classes + use_cfg_embeddings,
            hidden_size,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(0.02),
        )
        self.num_classes = num_classes
        self.dropout = drop

    def token_drop(self, labels, force_drop_ids=None) -> Array:
        if force_drop_ids is None:
            drop_ids = jrand.normal(key=randkey, shape=labels.shape[0])
        else:
            drop_ids = force_drop_ids == 1

        labels = jnp.where(drop_ids, self.num_classes, labels)

        return labels

    def __call__(self, labels, train: bool, force_drop_ids=None) -> Array:
        use_drop = self.dropout > 0
        if (train and use_drop) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        label_embeds = self.embedding_table(labels)

        return label_embeds


# self attention block
class SelfAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, rngs: nnx.Rngs, drop=0.0):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

        linear_init = nnx.initializers.xavier_uniform()
        linear_bias_init = nnx.initializers.constant(0)

        self.q_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.k_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.v_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs,
        )

        self.outproject = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(x_input)
        v = self.v_linear(x_input)

        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.attn_heads), (q, k, v)
        )

        qk = q @ jnp.matrix_transpose(k)
        attn_logits = qk / math.sqrt(self.head_dim)  # attention computation

        attn_score = nnx.softmax(attn_logits, axis=-1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, "b h l d -> b l (h d)")
        output = self.dropout(self.outproject(output))
        return output


class CrossAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, cond_dim, rngs: nnx.Rngs, drop=0.0):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

        linear_init = nnx.initializers.xavier_uniform()
        linear_bias_init = nnx.initializers.constant(0)

        self.q_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )

        self.k_linear = nnx.Linear(
            cond_dim,
            embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs,
        )
        self.v_linear = nnx.Linear(
            cond_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )

        self.outproject = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.Array, y_cond: Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(y_cond)
        v = self.v_linear(y_cond)

        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.attn_heads), (q, k, v)
        )

        qk = q @ jnp.matrix_transpose(k)
        attn_logits = qk / math.sqrt(self.head_dim)  # attention computation

        attn_score = nnx.softmax(attn_logits, axis=-1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, "b h l d -> b l (h d)")
        output = self.dropout(self.outproject(output))

        return output


class SimpleMLP(nnx.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class OutputMLP(nnx.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(
            embed_dim, patch_size[0] * patch_size[1] * out_channels, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class FinalMLP(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # linear_init = nnx.initializers.xavier_uniform()
        linear_init = nnx.initializers.constant(0)

        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=linear_init,
        )
        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=linear_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


###############
# DiT blocks_ #
###############


class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()

        # initializations
        linear_init = nnx.initializers.xavier_uniform()
        lnbias_init = nnx.initializers.constant(1)
        lnweight_init = nnx.initializers.constant(1)

        self.norm_1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )
        self.attention = SelfAttention(num_heads, hidden_size, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, scale_init=lnweight_init
        )

        self.adaln_linear = nnx.Linear(
            in_features=hidden_size,
            out_features=6 * hidden_size,
            use_bias=True,
            bias_init=zero_init,
            rngs=rngs,
            kernel_init=zero_init,
        )

        self.mlp_block = SimpleMLP(hidden_size)

    def __call__(self, x_img: Array, cond):

        cond = self.adaln_linear(nnx.silu(cond))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=1)
        )

        attn_mod_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))

        x = x_img + jnp.expand_dims(gate_msa, 1) * attn_mod_x
        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_mod_x = self.mlp_block(x)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_mod_x

        return x


class DiTBackbone(nnx.Module):
    def __init__(
        self,
        patch_size=(4, 4),
        in_channels=3,
        hidden_size=1024,
        depth=4,
        attn_heads=6,
    ):
        super().__init__()
        self.attn_heads = attn_heads

        dit_blocks = [
            DiTBlock(hidden_size, num_heads=attn_heads) for _ in tqdm(range(depth))
        ]
        self.dit_layers = nnx.Sequential(*dit_blocks)

        self.final_mlp = FinalMLP(hidden_size, patch_size[0], self.out_channels)

    def _unpatchify(self, x: Array) -> Array:

        bs, num_patches, patch_dim = x.shape
        H, W = self.patch_size  # Assume square patches
        in_channels = patch_dim // (H * W)
        height, width = config.img_size, config.img_size

        # Calculate the number of patches along each dimension
        num_patches_h = height // H
        num_patches_w = width // W

        # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
        x = x.reshape(bs, num_patches_h, num_patches_w, H, W, in_channels)

        # Transpose to (bs, num_patches_h, H, num_patches_w, W, in_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        reconstructed = x.reshape(
            bs, height, width, in_channels
        )  # Reshape to (bs, height, width, in_channels)

        return reconstructed

    def __call__(self, x: Array, y_cond: Array) -> Array:

        x = self.dit_layers(x, y_cond)
        x = self.final_mlp(x, y_cond)  # type: ignore
        x = self._unpatchify(x)

        return x

    def cfg_forward(self, x_img, y_cond, cfg_scale):
        half = x_img[: len(x_img) // 2]
        combined = jnp.concat([half, half], axis=0)
        model_out = self(combined, y_cond)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = jnp.split(eps, len(eps) // 2, axis=0)

        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concat([half_eps, half_eps], axis=0)
        cfg_out = jnp.concat([eps, rest], axis=1)

        return cfg_out


#####################
# Full Microdit model
####################
class MicroDiT(nnx.Module):
    def __init__(
        self,
        inchannels,
        patch_size,
        embed_dim,
        num_layers,
        attn_heads,
        mlp_dim,
        cond_embed_dim,
        dropout=0.0,
        patchmix_layers=2,
        rngs=rngs,
        num_classes=10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedder = PatchEmbed(
            rngs=rngs, patch_size=patch_size[0], in_chan=inchannels, embed_dim=embed_dim
        )
        self.num_patches = self.patch_embedder.num_patches

        # conditioning layers
        self.time_embedder = TimestepEmbedder(embed_dim)
        self.label_embedder = LabelEmbedder(
            num_classes=num_classes, hidden_size=embed_dim, drop=dropout
        )
        self.cond_attention = CrossAttention(
            attn_heads, embed_dim, cond_embed_dim, rngs=rngs
        )
        self.cond_mlp = SimpleMLP(embed_dim)

        self.linear = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)

        self.backbone = DiTBackbone(
            patch_size=(2, 2),
            in_channels=3,
            hidden_size=embed_dim,
            depth=num_layers,
            attn_heads=attn_heads,
        )

    def __call__(self, x: Array, t: Array, y_cap: Array, mask=None):
        bsize, channels, height, width = x.shape
        psize_h, psize_w = self.patch_size

        x = self.patch_embedder(x)

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, height // psize_h)

        pos_embed = jnp.expand_dims(pos_embed, axis=0)
        pos_embed = jnp.broadcast_to(
            pos_embed, (bsize, pos_embed.shape[1], pos_embed.shape[2])
        )
        pos_embed = jnp.reshape(
            pos_embed, shape=(bsize, self.num_patches, self.embed_dim)
        )
        x = jnp.reshape(x, shape=(bsize, self.num_patches, self.embed_dim))

        x = x + pos_embed

        cond_embed = self.label_embedder(y_cap, train=True)
        cond_embed_unsqueeze = jnp.expand_dims(cond_embed, axis=1)

        time_embed = self.time_embedder(t)
        time_embed_unsqueeze = jnp.expand_dims(time_embed, axis=1)

        mha_out = self.cond_attention(time_embed_unsqueeze, cond_embed_unsqueeze)  #
        mha_out = mha_out.squeeze(1)

        mlp_out = self.cond_mlp(mha_out)

        # # pooling the conditions
        pool_out = self.pool_mlp(jnp.expand_dims(mlp_out, axis=2))

        pool_out = jnp.expand_dims((pool_out + time_embed), axis=1)

        cond_signal = jnp.expand_dims(self.linear(mlp_out), axis=1)
        cond_signal = jnp.broadcast_to((cond_signal + pool_out), shape=(x.shape))
        x = x + cond_signal

        mlp_out_us = jnp.expand_dims(mlp_out, axis=1)  # unqueezed mlp output

        cond = jnp.broadcast_to((mlp_out_us + pool_out), shape=(x.shape))

        x = x + cond
        x = self.ditbackbone(x, cond_embed)

        return x

    def sample(self, z_latent, cond, sample_steps=50, cfg=2.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for i in tqdm(range(sample_steps, 0, -1)):
            t = i / sample_steps
            t = jnp.array([t] * b_size).astype(z_latent.dtype)

            vc = self(z_latent, t, cond, None)
            null_cond = jnp.zeros_like(cond)
            vu = self.__call__(z_latent, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z_latent = z_latent - dt * vc
            images.append(z_latent)

        return images  # [-1]


# rectifed flow forward pass, loss, and smapling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array, mask) -> Array:

        b_size = x_input.shape[0]  # batch_sie
        rand_t = None

        if self.sigln:
            rand = jrand.normal(randkey, (b_size,))  # .to_device(x_input.device)
            rand_t = nnx.sigmoid(rand)
        else:
            rand_t = jrand.normal(randkey, (b_size,))  # .to_device(x_input.device)

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)])

        z_noise = jrand.normal(
            randkey, x_input.shape
        )  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise
        # print(z_noise_t)
        v_thetha = self.model(z_noise_t, rand_t, cond, mask)

        mean_dim = list(
            range(1, len(x_input.shape))
        )  # across all dimensions except the batch dim
        print(
            f"z_noise {z_noise.shape}, vtheta {v_thetha.shape}, x_input {x_input.shape}"
        )

        mean_square = (z_noise - x_input - v_thetha) ** 2  # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss

        return jnp.mean(batchwise_mse_loss)

    def sample(
        self,
        input_noise: jax.Array,
        cond,
        zero_cond=None,
        sample_steps: int = 50,
        cfg=2.0,
    ) -> List[jax.Array]:

        batch_size = input_noise.shape[0]

        # array reciprocal of sampling steps
        d_steps = 1.0 / sample_steps

        d_steps = jnp.array([d_steps] * batch_size)  # .to_device(input_noise.device)
        steps_dim = [1] * len(input_noise.shape[1:])
        d_steps = d_steps.reshape((batch_size, *steps_dim))

        images = [input_noise]  # noise sequence

        for t_step in tqdm(range(sample_steps)):

            genstep = t_step / sample_steps  # current step

            genstep_batched = jnp.array(
                [genstep] * batch_size
            )  # .to_device(input_noise.device)

            cond_output = self.model(
                input_noise, genstep_batched, cond
            )  # get model output for step

            if zero_cond is not None:
                # output for zero conditioning
                uncond_output = self.model(input_noise, genstep_batched, zero_cond)
                cond_output = uncond_output + cfg * (cond_output - uncond_output)

            input_noise = input_noise - d_steps * cond_output

            images.append(input_noise)

        return images


microdit = MicroDiT(
    inchannels=3,
    patch_size=(2, 2),
    embed_dim=1024,
    num_layers=16,
    attn_heads=16,
    mlp_dim=4 * 1024,
    cond_embed_dim=1024,
)

rf_engine = RectFlowWrapper(microdit)
graph, state = nnx.split(rf_engine)

n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.3f}M")
