import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model'
    )
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped '
              'into the directory as json')
    )
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation'
    )
    parser.add_argument(
        '--show', action='store_true', help='show prediction results'
    )
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the work_dir/timestamp/show_dir'
    )
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)'
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation'
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_config(args):
    """Setup and configure the testing configuration."""
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def setup_work_dir(cfg, args):
    """Setup the working directory."""
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0]
        )
    return cfg


def setup_visualization(cfg, args):
    """Setup visualization hooks if requested."""
    if not (args.show or args.show_dir):
        return cfg

    default_hooks = cfg.default_hooks
    if 'visualization' not in default_hooks:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks. '
            'refer to usage "visualization=dict(type=\'VisualizationHook\')"'
        )

    visualization_hook = default_hooks['visualization']
    # Turn on visualization
    visualization_hook['draw'] = True

    if args.show:
        visualization_hook['show'] = True
        visualization_hook['wait_time'] = args.wait_time

    if args.show_dir:
        visualizer = cfg.visualizer
        visualizer['save_dir'] = args.show_dir

    return cfg


def setup_tta(cfg, args):
    """Setup test time augmentation if requested."""
    if not args.tta:
        return cfg

    cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    cfg.tta_model.module = cfg.model
    cfg.model = cfg.tta_model

    return cfg


def setup_output_dir(cfg, args):
    """Setup output directory for evaluation results."""
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    return cfg


def main():
    """Main testing function."""
    args = parse_args()

    # Setup configuration
    cfg = setup_config(args)
    cfg = setup_work_dir(cfg, args)
    cfg.load_from = args.checkpoint

    # Setup optional features
    cfg = setup_visualization(cfg, args)
    cfg = setup_tta(cfg, args)
    cfg = setup_output_dir(cfg, args)

    # Build and run tester
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()