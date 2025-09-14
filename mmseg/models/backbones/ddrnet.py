# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize, SpatialAttention, LiteMLA
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.backbones.batchnorm import SynchronizedBatchNorm2d


@MODELS.register_module()
class DDRNet(BaseModule):
    """DDRNet backbone for real-time semantic segmentation.

    Implementation of `Deep Dual-resolution Networks for Real-time and Accurate
    Semantic Segmentation of Road Scenes <http://arxiv.org/abs/2101.06085>`_.
    Modified from https://github.com/ydhongHIT/DDRNet.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels (int): Base channels of the DDRNet. Default: 32.
        ppm_channels (int): Channels of the DAPPM (PPM variant) module. Default: 128.
        align_corners (bool): `align_corners` for `F.interpolate`. Default: False.
        norm_cfg (OptConfigType): Config for normalization layers.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (OptConfigType): Config for activation layers.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (OptConfigType): Initialization config. Default: None.
    """

    def __init__(
            self,
            in_channels: int = 3,
            channels: int = 32,
            ppm_channels: int = 128,
            align_corners: bool = False,
            norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
            act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
            init_cfg: OptConfigType = None
    ):
        super().__init__(init_cfg)

        # Core configurations
        self.in_channels = in_channels
        self.base_channels = channels
        self.ppm_channels = ppm_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.relu = nn.ReLU()  # Reusable ReLU activation

        # Initialize network components
        self._build_stem()
        self._build_dual_branches()
        self._build_bilateral_fusion()
        self._build_strip_modules()
        self._build_attention_modules()
        self._build_spp_module()

    def _build_stem(self) -> None:
        """Build the stem layer (stage 0-2) to downsample input and extract initial features."""
        stem_layers = [
            # Initial 3x3 convs with stride=2 (downsample by 4x)
            ConvModule(
                self.in_channels, self.base_channels,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
            ),
            ConvModule(
                self.base_channels, self.base_channels,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
            ),
            # BasicBlock layers (no downsample)
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels,
                planes=self.base_channels,
                num_blocks=2
            ),
            self.relu,
            # BasicBlock layer with stride=2 (downsample by 2x)
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels,
                planes=self.base_channels * 2,
                num_blocks=2,
                stride=2
            ),
            self.relu
        ]
        self.stem = nn.Sequential(*stem_layers)

    def _build_dual_branches(self) -> None:
        """Build dual branches: Context (low-res, high-semantic) and Spatial (high-res, low-semantic)."""
        # Context branch (3 stages: BasicBlock x2 → BasicBlock x2 → Bottleneck x1)
        self.context_branch = ModuleList([
            # Stage 3: in=64 (32*2), out=128 (32*4), stride=2
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels * 2,
                planes=self.base_channels * 4,
                num_blocks=2,
                stride=2
            ),
            # Stage 4: in=128, out=256 (32*8), stride=2
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels * 4,
                planes=self.base_channels * 8,
                num_blocks=2,
                stride=2
            ),
            # Stage 5: in=256, out=256 (Bottleneck expansion=4 → 32*8=256), stride=1
            self._make_layer(
                block=Bottleneck,
                inplanes=self.base_channels * 8,
                planes=self.base_channels * 8,  # expansion=4 → 8*4=32? No: planes*expansion = 8*32*4=1024?
                num_blocks=1,
                stride=1
            )
        ])

        # Spatial branch (3 stages: all BasicBlock/Bottleneck with no downsample)
        self.spatial_branch = ModuleList([
            # Stage 3: in=64, out=64, stride=1
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels * 2,
                planes=self.base_channels * 2,
                num_blocks=2,
                stride=1
            ),
            # Stage 4: in=64, out=64, stride=1
            self._make_layer(
                block=BasicBlock,
                inplanes=self.base_channels * 2,
                planes=self.base_channels * 2,
                num_blocks=2,
                stride=1
            ),
            # Stage 5: in=64, out=64 (Bottleneck expansion=4 → 2*32*4=256? No: planes*expansion=2*32*4=256)
            self._make_layer(
                block=Bottleneck,
                inplanes=self.base_channels * 2,
                planes=self.base_channels * 2,
                num_blocks=1,
                stride=1
            )
        ])

    def _build_bilateral_fusion(self) -> None:
        """Build bilateral fusion modules (compression + downsampling) for cross-branch interaction."""
        # Fusion for Stage 3: Context (128) → compress to 64; Spatial (64) → downsample to 128
        self.compression_stage3 = ConvModule(
            in_channels=self.base_channels * 4,  # Context output: 32*4=128
            out_channels=self.base_channels * 2,  # Compress to 64 (match Spatial channel)
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None  # Activation applied later in forward
        )
        self.downsample_stage3 = ConvModule(
            in_channels=self.base_channels * 2,  # Spatial output: 64
            out_channels=self.base_channels * 4,  # Downsample to 128 (match Context channel)
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None
        )

        # Fusion for Stage 4: Context (256) → compress to 64; Spatial (64) → downsample to 256
        self.compression_stage4 = ConvModule(
            in_channels=self.base_channels * 8,  # Context output: 32*8=256
            out_channels=self.base_channels * 2,  # Compress to 64
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None
        )
        self.downsample_stage4 = nn.Sequential(
            ConvModule(
                in_channels=self.base_channels * 2,
                out_channels=self.base_channels * 4,  # 64 → 128
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                in_channels=self.base_channels * 4,
                out_channels=self.base_channels * 8,  # 128 → 256
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        )

    def _build_strip_modules(self) -> None:
        """Build reusable "strip" modules (multi-scale conv branches) for Spatial branch refinement.

        Two variants:
        - Strip Small: For 64-channel inputs (Stage 3/4 Spatial branch)
        - Strip Large: For 128-channel inputs (Stage 5 Spatial branch)
        """
        BatchNorm = SynchronizedBatchNorm2d

        # ------------------------------
        # Strip Small (input: 64 channels)
        # ------------------------------
        strip_small_in = 64
        strip_small_mid = strip_small_in // 4  # 16
        strip_small_branch = strip_small_mid // 2  # 8
        self.strip_small = nn.ModuleDict({
            "conv_reduce": nn.Conv2d(strip_small_in, strip_small_mid, kernel_size=1),
            "bn_reduce": BatchNorm(strip_small_mid),
            "branches": nn.ModuleList([
                nn.Conv2d(strip_small_mid, strip_small_branch, kernel_size=(1, 9), padding=(0, 4)),  # Horizontal
                nn.Conv2d(strip_small_mid, strip_small_branch, kernel_size=(9, 1), padding=(4, 0)),  # Vertical
                nn.Conv2d(strip_small_mid, strip_small_branch, kernel_size=3, padding=1),  # 3x3
                nn.Conv2d(strip_small_mid, strip_small_branch, kernel_size=3, padding=2, dilation=2)  # Dilated 3x3
            ]),
            "bn_fuse": BatchNorm(strip_small_mid * 2),  # 16*2=32 (4 branches × 8 = 32)
            "conv_project": ConvModule(
                in_channels=strip_small_mid * 2,
                out_channels=strip_small_in,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        })

        # ------------------------------
        # Strip Large (input: 128 channels)
        # ------------------------------
        strip_large_in = 128
        strip_large_mid = strip_large_in // 4  # 32
        strip_large_branch = strip_large_mid // 2  # 16
        self.strip_large = nn.ModuleDict({
            "conv_reduce": nn.Conv2d(strip_large_in, strip_large_mid, kernel_size=1),
            "bn_reduce": BatchNorm(strip_large_mid),
            "branches": nn.ModuleList([
                nn.Conv2d(strip_large_mid, strip_large_branch, kernel_size=(1, 9), padding=(0, 4)),
                nn.Conv2d(strip_large_mid, strip_large_branch, kernel_size=(9, 1), padding=(4, 0)),
                nn.Conv2d(strip_large_mid, strip_large_branch, kernel_size=3, padding=1),
                nn.Conv2d(strip_large_mid, strip_large_branch, kernel_size=3, padding=2, dilation=2)
            ]),
            "bn_fuse": BatchNorm(strip_large_mid * 2),  # 32*2=64 (4 branches ×16=64)
            "conv_project": ConvModule(
                in_channels=strip_large_mid * 2,
                out_channels=strip_large_in,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        })

    def _build_attention_modules(self) -> None:
        """Build cross-attention and spatial attention modules for feature fusion."""
        # Cross-branch linear attention (LiteMLA)
        self.cross_attn_context3 = LiteMLA(in_channels=128, out_channels=128)  # Stage3 Context (128)
        self.cross_attn_spatial3 = LiteMLA(in_channels=64, out_channels=64)  # Stage3 Spatial (64)
        self.cross_attn_context4 = LiteMLA(in_channels=256, out_channels=256)  # Stage4 Context (256)
        self.cross_attn_spatial4 = LiteMLA(in_channels=64, out_channels=64)  # Stage4 Spatial (64)

        # Spatial attention (channel-wise pooling + conv)
        self.spatial_attn1 = SpatialAttention()  # For Stage3 downsampled Spatial
        self.spatial_attn2 = SpatialAttention()  # For Stage3 compressed Context
        self.spatial_attn3 = SpatialAttention()  # For Stage4 downsampled Spatial
        self.spatial_attn4 = SpatialAttention()  # For Stage4 compressed Context

    def _build_spp_module(self) -> None:
        """Build DAPPM (Dilated Atrous Pyramid Pooling Module) for context branch."""
        self.spp = DAPPM(
            in_channels=self.base_channels * 16,  # Context Stage5 output: 32*16=512?
            out_channels=self.ppm_channels,  # 128
            mid_channels=self.base_channels * 4,  # 128
            num_scales=5
        )

    def _make_layer(
            self,
            block: type[BasicBlock | Bottleneck],
            inplanes: int,
            planes: int,
            num_blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple residual blocks.

        Args:
            block: Type of residual block (BasicBlock or Bottleneck).
            inplanes: Number of input channels.
            planes: Number of output channels (per block).
            num_blocks: Number of residual blocks in the layer.
            stride: Stride of the first block (for downsampling).

        Returns:
            Sequential layer containing residual blocks.
        """
        # Downsample if input/output channels mismatch or stride != 1
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1]
            )

        # Build residual blocks
        layers = []
        # First block (may include downsample)
        layers.append(
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        inplanes = planes * block.expansion

        # Remaining blocks (no downsample)
        for i in range(1, num_blocks):
            # Disable activation for the last block (activated later in forward)
            act_cfg_out = None if i == num_blocks - 1 else self.act_cfg
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=act_cfg_out
                )
            )

        return nn.Sequential(*layers)

    def _apply_strip_module(
            self,
            strip_module: nn.ModuleDict,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Apply a pre-built strip module to refine features.

        Args:
            strip_module: Strip module (small or large) from `self.strip_small`/`self.strip_large`.
            x: Input tensor to refine.

        Returns:
            Refined feature tensor.
        """
        # Channel reduction (1x1 conv)
        x = strip_module["conv_reduce"](x)
        x = strip_module["bn_reduce"](x)
        x = self.relu(x)

        # Multi-scale conv branches
        branch_outputs = [branch(x) for branch in strip_module["branches"]]
        x = torch.cat(branch_outputs, dim=1)

        # Fusion and projection
        x = strip_module["bn_fuse"](x)
        x = self