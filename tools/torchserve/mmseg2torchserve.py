from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from mmengine import Config
from mmengine.utils import mkdir_or_exist

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None


def mmseg2torchserve(
        config_file: str,
        checkpoint_file: str,
        output_folder: str,
        model_name: str,
        model_version: str = '1.0',
        force: bool = False,
):
    """Converts mmsegmentation model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file: In MMSegmentation config format.
        checkpoint_file: In MMSegmentation checkpoint format.
        output_folder: Folder where `{model_name}.mar` will be created.
        model_name: Name for the model archive file.
        model_version: Model's version, defaults to '1.0'.
        force: If True, overwrite existing model archive.
    """
    mkdir_or_exist(output_folder)

    # Load and validate config
    config = Config.fromfile(config_file)

    with TemporaryDirectory() as tmp_dir:
        # Save config to temporary directory
        config_path = Path(tmp_dir) / 'config.py'
        config.dump(str(config_path))

        # Prepare packaging arguments
        package_args = _create_package_args(
            config_path=str(config_path),
            checkpoint_file=checkpoint_file,
            output_folder=output_folder,
            model_name=model_name or Path(checkpoint_file).stem,
            model_version=model_version,
            force=force
        )

        # Generate manifest and package model
        manifest = ModelExportUtils.generate_manifest_json(package_args)
        package_model(package_args, manifest)


def _create_package_args(
        config_path: str,
        checkpoint_file: str,
        output_folder: str,
        model_name: str,
        model_version: str,
        force: bool
) -> Namespace:
    """Create packaging arguments for model archiver."""
    handler_path = Path(__file__).parent / 'mmseg_handler.py'

    return Namespace(
        model_file=config_path,
        serialized_file=checkpoint_file,
        handler=str(handler_path),
        model_name=model_name,
        version=model_version,
        export_path=output_folder,
        force=force,
        requirements_file=None,
        extra_files=None,
        runtime='python',
        archive_format='default'
    )


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(
        description='Convert mmseg models to TorchServe `.mar` format.'
    )
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Folder where `{model_name}.mar` will be created.'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name for the model archive file. '
             'If None, uses checkpoint filename stem.'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0',
        help='Model version number.'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Overwrite existing model archive.'
    )

    return parser.parse_args()


def main():
    """Main function to convert MMSeg model to TorchServe format."""
    args = parse_args()

    if package_model is None:
        raise ImportError(
            '`torch-model-archiver` is required. '
            'Try: pip install torch-model-archiver'
        )

    mmseg2torchserve(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        output_folder=args.output_folder,
        model_name=args.model_name,
        model_version=args.model_version,
        force=args.force
    )


if __name__ == '__main__':
    main()