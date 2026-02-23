import numpy as np

import jittor as jt
import jittor.nn as nn


def _trunc_normal_(var, std=0.02):
    shape = tuple(var.shape)
    vals = np.random.normal(loc=0.0, scale=std, size=shape)
    low, high = -2 * std, 2 * std
    invalid = (vals < low) | (vals > high)
    while np.any(invalid):
        vals[invalid] = np.random.normal(loc=0.0, scale=std, size=int(invalid.sum()))
        invalid = (vals < low) | (vals > high)
    vals = vals.astype("float32")
    try:
        var.assign(jt.array(vals))
    except Exception:
        try:
            var.update(jt.array(vals))
        except Exception:
            pass


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def execute(self, x):
        if self.drop_prob == 0.0:
            return x
        training = getattr(self, "training", None)
        if training is False:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (jt.rand(shape) < keep_prob).float()
        return x * mask / keep_prob


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        flatten=True,
        norm_layer=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flatten = flatten
        self.proj = nn.Conv(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def execute(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.transpose(0, 2, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        _trunc_normal_(self.fc1.weight, std=0.02)
        if self.fc1.bias is not None:
            try:
                self.fc1.bias.assign(jt.zeros(self.fc1.bias.shape))
            except Exception:
                self.fc1.bias.update(jt.zeros(self.fc1.bias.shape))
        _trunc_normal_(self.fc2.weight, std=0.02)
        if self.fc2.bias is not None:
            try:
                self.fc2.bias.assign(jt.zeros(self.fc2.bias.shape))
            except Exception:
                self.fc2.bias.update(jt.zeros(self.fc2.bias.shape))

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._init_weights()

    def _init_weights(self):
        _trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            try:
                self.qkv.bias.assign(jt.zeros(self.qkv.bias.shape))
            except Exception:
                self.qkv.bias.update(jt.zeros(self.qkv.bias.shape))
        _trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            try:
                self.proj.bias.assign(jt.zeros(self.proj.bias.shape))
            except Exception:
                self.proj.bias.update(jt.zeros(self.proj.bias.shape))

    def execute(self, x):
        bsz, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = jt.matmul(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(bsz, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self._init_norms()

    def _init_norms(self):
        for norm in (self.norm1, self.norm2):
            try:
                norm.weight.assign(jt.ones(norm.weight.shape))
            except Exception:
                try:
                    norm.weight.update(jt.ones(norm.weight.shape))
                except Exception:
                    pass
            try:
                norm.bias.assign(jt.zeros(norm.bias.shape))
            except Exception:
                try:
                    norm.bias.update(jt.zeros(norm.bias.shape))
                except Exception:
                    pass

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            flatten=True,
            norm_layer=None,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        drop_rates = np.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.feature_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)
        try:
            self.norm.weight.assign(jt.ones(self.norm.weight.shape))
        except Exception:
            try:
                self.norm.weight.update(jt.ones(self.norm.weight.shape))
            except Exception:
                pass
        try:
            self.norm.bias.assign(jt.zeros(self.norm.bias.shape))
        except Exception:
            try:
                self.norm.bias.update(jt.zeros(self.norm.bias.shape))
            except Exception:
                pass

    def execute(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = jt.concat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]
