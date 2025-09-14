# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from torch import nn
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmseg.models import build_segmentor

torch.manual_seed(3)


def digit_version(version_str: str) -> List[int]:
    """Convert version string to a list of integers for comparison."""
    digit_version = []
    for part in version_str.split('.'):
        if part.isdigit():
            digit_version.append(int(part))
        elif 'rc' in part:
            rc_parts = part.split('rc')
            digit_version.append(int(rc_parts[0]) - 1)
            digit_version.append(int(rc_parts[1]))
    return digit_version


def check_torch_version() -> None:
    """Check if the installed PyTorch version meets the minimum requirement."""
    torch_minimum_version = '1.8.0'
    torch_version = digit_version(torch.__version__)

    assert torch_version >= digit_version(torch_minimum_version), \
        f'Torch=={torch.__version__} is not supported for converting to ' \
        f'torchscript. Please install pytorch>={torch_minimum_version}.'


def _convert_batchnorm(module: nn.Module) -> nn.Module:
    """Convert SyncBatchNorm to BatchNorm2d in a recursive manner."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats
        )

        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    # Recursively convert child modules
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))

    del module
    return module_output


def _demo_mm_inputs(input_shape: Tuple[int, int, int, int],
                    num_classes: int) -> Dict[str, Any]:
    """Create dummy inputs for MMSegmentation models.

    Args:
        input_shape: Tuple of (N, C, H, W)
        num_classes: Number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0,
        high=num_classes - 1,
        size=(N, 1, H, W)
    ).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    return {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }


def pytorch2libtorch(model: nn.Module,
                     input_shape: Tuple[int, int, int, int],
                     output_file: str = 'tmp.pt',
                     show: bool = False,
                     verify: bool = False) -> None:
    """Export PyTorch model to TorchScript.

    Args:
        model: PyTorch model to export
        input_shape: Input shape (N, C, H, W)
        output_file: Path to save the TorchScript model
        show: Whether to print the computation graph
        verify: Whether to verify the exported model
    """
    # Determine number of classes
    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    # Prepare dummy inputs
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)
    imgs = mm_inputs.pop('imgs')

    # Export model
    model.forward = model.forward_dummy  # type: ignore
    model.eval()

    traced_model = torch.jit.trace(
        model,
        example_inputs=imgs,
        check_trace=verify,
    )

    if show:
        print(traced_model.graph)

    traced_model.save(output_file)
    print(f'Successfully exported TorchScript model: {output_file}')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert MMSeg to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--show', action='store_true', help='show TorchScript graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the TorchScript model')
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size (height, width)')

    return parser.parse_args()


def main() -> None:
    """Main function for model conversion."""
    args = parse_args()
    check_torch_version()

    # Process input shape
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)  # type: ignore
    else:
        raise ValueError('Invalid input shape')

    # Load configuration and build model
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    segmentor = build_segmentor(
        cfg.model,
        train_cfg=None,
        test_cfg=cfg.get('test_cfg')
    )

    # Convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    # Convert to TorchScript
    pytorch2libtorch(
        segmentor,
        input_shape=input_shape,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify
    )


if __name__ == '__main__':
    main()
