from mmengine.model import BaseModule, ModuleList
from functools import partial
from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==================== UTILITY FUNCTIONS ====================
def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    """Calculate same padding for convolution layers."""
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def val2list(x: list or tuple or any, repeat_time: int = 1) -> list:
    """Convert value to list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """Convert value to tuple with minimum length."""
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)

def build_kwargs_from_config(config: dict, target_func: callable) -> dict:
    """Build kwargs dictionary from config for target function."""
    valid_keys = list(signature(target_func).parameters)
    return {key: config[key] for key in config if key in valid_keys}

# ==================== ACTIVATION REGISTRY ====================
REGISTERED_ACT_DICT = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}

def build_act(name: str, **kwargs) -> nn.Module or None:
    """Build activation function from registry."""
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    return None

# ==================== NORMALIZATION LAYERS ====================
class LayerNorm2d(nn.LayerNorm):
    """2D Layer Normalization implementation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

REGISTERED_NORM_DICT = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}

def build_norm(name: str = "bn2d", num_features: int = None, **kwargs) -> nn.Module or None:
    """Build normalization layer from registry."""
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
        
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    return None

# ==================== BASIC CONVOLUTION LAYER ====================
class ConvLayer(nn.Module):
    """Basic convolution layer with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "relu",
    ):
        super().__init__()
        
        padding = get_same_padding(kernel_size) * dilation
        
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, use_bias
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

# ==================== ATTENTION MODULES ====================
class DilateAttention(BaseModule):
    """Dilated Attention implementation."""
    
    def __init__(
        self, 
        head_dim: int, 
        qk_scale: float = None, 
        attn_drop: float = 0, 
        kernel_size: int = 3, 
        dilation: int = 1
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, d, H, W = q.shape
        
        # Reshape query
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1, H*W]).permute(0, 1, 4, 3, 2)
        
        # Process key with unfolding
        k = self.unfold(k).reshape(
            [B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]
        ).permute(0, 1, 4, 2, 3)
        
        # Compute attention
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Process value with unfolding
        v = self.unfold(v).reshape(
            [B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]
        ).permute(0, 1, 4, 3, 2)
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilateLocalAttention(BaseModule):
    """Multi-scale Dilated Local Attention implementation."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        kernel_size: int = 3,
        dilation: list = [1, 2],
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        
        assert num_heads % self.num_dilation == 0, \
            f"num_heads {num_heads} must be divisible by num_dilation {self.num_dilation}"
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i]) 
            for i in range(self.num_dilation)
        ])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W)
        qkv = qkv.permute(2, 1, 0, 3, 4, 5)  # num_dilation, 3, B, C//num_dilation, H, W
        
        # Process each dilation branch
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W)
        x = x.permute(1, 0, 3, 4, 2)  # num_dilation, B, H, W, C//num_dilation
        
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        
        # Combine and project
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LiteMLA(nn.Module):
    """Lightweight Multi-scale Linear Attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = None,
        heads_ratio: float = 1.0,
        dim: int = 1,
        use_bias: bool = False,
        norm: tuple = (None, "bn2d"),
        act_func: tuple = (None, None),
        kernel_func: str = "relu",
        scales: tuple = (3, 3),
        eps: float = 1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels, 3 * total_dim, 1, 
            use_bias=use_bias[0], norm=norm[0], act_func=act_func[0]
        )

        # Multi-scale aggregation
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dim, 3 * total_dim, scale,
                    padding=get_same_padding(scale),
                    groups=3 * total_dim,
                    bias=use_bias[0],
                ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
            for scale in scales
        ])

        # Additional convolution branches
        self.conv_branches = self._build_conv_branches(3 * total_dim, heads, use_bias[0])
        self.proj = ConvLayer(
            total_dim * (1 + len(scales) + len(self.conv_branches)), out_channels, 1,
            use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

    def _build_conv_branches(self, in_channels: int, heads: int, use_bias: bool) -> nn.ModuleList:
        """Build additional convolution branches."""
        branches = nn.ModuleList()
        
        # Branch 1: Horizontal strip
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 9), padding=(0, 4), groups=3 * heads),
            nn.Conv2d(in_channels, in_channels, 1, groups=3 * heads, bias=use_bias)
        ))
        
        # Branch 2: Vertical strip
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (9, 1), padding=(4, 0), groups=3 * heads),
            nn.Conv2d(in_channels, in_channels, 1, groups=3 * heads, bias=use_bias)
        ))
        
        # Branch 3: 3x3 convolution
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=3 * heads),
            nn.Conv2d(in_channels, in_channels, 1, groups=3 * heads, bias=use_bias)
        ))
        
        # Branch 4: Dilated convolution
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=3 * heads),
            nn.Conv2d(in_channels, in_channels, 1, groups=3 * heads, bias=use_bias)
        ))
        
        return branches

    def _relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        """Linear attention with ReLU activation."""
        B, _, H, W = qkv.size()
        
        if qkv.dtype in [torch.float16, torch.bfloat16]:
            qkv = qkv.float()

        qkv = qkv.reshape(B, -1, 3 * self.dim, H * W)
        q, k, v = qkv.chunk(3, dim=2)

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        
        if out.dtype == torch.bfloat16:
            out = out.float()
            
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(B, -1, H, W)

    def _relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        """Quadratic attention with ReLU activation."""
        B, _, H, W = qkv.size()
        
        qkv = qkv.reshape(B, -1, 3 * self.dim, H * W)
        q, k, v = qkv.chunk(3, dim=2)

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)
        original_dtype = att_map.dtype
        
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
            
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(original_dtype)
        
        out = torch.matmul(v, att_map)
        return out.reshape(B, -1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate multi-scale QKV
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]

        # Apply multi-scale aggregation
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))

        # Apply additional convolution branches
        for branch in self.conv_branches:
            multi_scale_qkv.append(branch(qkv))

        # Concatenate all features
        qkv = torch.cat(multi_scale_qkv, dim=1)
        H, W = qkv.shape[-2:]

        # Select attention mechanism based on feature size
        if H * W > self.dim:
            out = self._relu_linear_att(qkv)
        else:
            out = self._relu_quadratic_att(qkv)
            
        return self.proj(out)

# ==================== TRANSFORMER COMPONENTS ====================
class PreNorm(nn.Module):
    """Pre-normalization wrapper for transformer blocks."""
    
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block with multiple layers."""
    
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ==================== MOBILEViT COMPONENTS ====================
class MV2Block(nn.Module):
    """MobileNetV2 inverted residual block."""
    
    def __init__(self, inp: int, oup: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileViTBlock(nn.Module):
    """MobileViT block combining convolution and transformer."""
    
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        channel: int, 
        kernel_size: int, 
        patch_size: tuple, 
        mlp_dim: int, 
        dropout: float = 0.
    ):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', 
                     h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

# ==================== SPATIAL ATTENTION ====================
class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# ==================== HELPER CONVOLUTION FUNCTIONS ====================
def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    """1x1 convolution with batch normalization and SiLU activation."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp: int, oup: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """NxN convolution with batch normalization and SiLU activation."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
