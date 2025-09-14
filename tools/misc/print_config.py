# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from pathlib import Path

from mmengine import Config, DictAction

from mmseg.apis import init_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--graph',
        action='store_true',
        help='print the models graph'
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in '
             'the used config, the key-value pair in xxx=yyy format will be '
             'merged into config file. If the value to be overwritten is a '
             'list, it should be like key="[a,b]" or key=a,b It also allows '
             'nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that '
             'the quotation marks are necessary and that no white space is '
             'allowed.'
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

    args = parser.parse_args()
    _validate_and_process_args(args)

    return args


def _validate_and_process_args(args):
    """Validate and process command line arguments."""
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified. '
            '--options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.'
        )

    if args.options:
        warnings.warn(
            '--options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.',
            DeprecationWarning,
            stacklevel=2
        )
        args.cfg_options = args.options


def load_and_merge_config(config_path, cfg_options=None):
    """Load and merge configuration."""
    cfg = Config.fromfile(config_path)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg


def save_config(cfg, output_path='example.py'):
    """Save configuration to file."""
    cfg.dump(output_path)
    print(f'Config saved to: {output_path}')


def save_model_graph(model, output_path='example-graph.txt'):
    """Save model graph to file."""
    model_str = str(model)
    with open(output_path, 'w') as f:
        f.write(model_str)
    print(f'Model graph saved to: {output_path}')
    return model_str


def main():
    """Main function to print and save config and model graph."""
    args = parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load and process config
    cfg = load_and_merge_config(args.config, args.cfg_options)

    # Print and save config
    print(f'Config:\n{cfg.pretty_text}')
    save_config(cfg)

    # Process model graph if requested
    if args.graph:
        model = init_model(args.config, device='cpu')
        model_graph = save_model_graph(model)
        print(f'Model graph:\n{model_graph}')


if __name__ == '__main__':
    main()