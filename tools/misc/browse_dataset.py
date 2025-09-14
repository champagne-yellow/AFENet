# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path

from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS


def parse_args():
    """Parse command line arguments for dataset browsing."""
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it'
    )
    parser.add_argument(
        '--not-show',
        default=False,
        action='store_true',
        help='Do not show images interactively'
    )
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value '
             'to be overwritten is a list, it should be like key="[a,b]" or '
             'key=a,b It also allows nested list/tuple values, e.g. '
             'key="[(a,b),(c,d)]" Note that the quotation marks are necessary '
             'and that no white space is allowed.'
    )
    return parser.parse_args()


def setup_config(args):
    """Setup and validate configuration."""
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def setup_visualizer(cfg, output_dir):
    """Setup visualizer with output directory."""
    if output_dir:
        cfg.visualizer['save_dir'] = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return VISUALIZERS.build(cfg.visualizer)


def process_dataset_item(visualizer, item, show=True, wait_time=2):
    """Process and visualize a single dataset item."""
    img = item['inputs'].permute(1, 2, 0).numpy()
    data_sample = item['data_samples'].numpy()
    img_path = osp.basename(item['data_samples'].img_path)

    # Convert BGR to RGB
    img = img[..., [2, 1, 0]]

    visualizer.add_datasample(
        osp.basename(img_path),
        img,
        data_sample,
        show=show,
        wait_time=wait_time
    )


def main():
    """Main function to browse dataset."""
    args = parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Setup configuration
    cfg = setup_config(args)

    # Register all modules in mmseg into the registries
    init_default_scope('mmseg')

    # Build dataset
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    # Setup visualizer
    visualizer = setup_visualizer(cfg, args.output_dir)
    visualizer.dataset_meta = dataset.METAINFO

    # Process each dataset item
    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        process_dataset_item(
            visualizer,
            item,
            show=not args.not_show,
            wait_time=args.show_interval
        )
        progress_bar.update()


if __name__ == '__main__':
    main()