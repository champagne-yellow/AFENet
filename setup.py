import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def readme():
    """Read the content of README.md file."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version():
    """Get version number from version file."""
    version_file = 'mmseg/version.py'
    with open(version_file) as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def _parse_line(line):
    """Parse a single line from requirements file."""
    if line.startswith('-r '):
        # Allow specifying requirements in other files
        target = line.split(' ')[1]
        yield from _parse_require_file(target)
    else:
        info = {'line': line}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        else:
            # Remove versioning from the package
            import re
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]

            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    # Handle platform specific dependencies
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest
                info['version'] = (op, version)
        yield info


def _parse_require_file(fpath):
    """Parse requirements file."""
    with open(fpath) as f:
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith('#'):
                yield from _parse_line(line)


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse package dependencies listed in a requirements file.

    Args:
        fname (str): Path to requirements file
        with_version (bool): If True include version specs, default True

    Returns:
        List[str]: List of requirements items
    """
    if not osp.exists(fname):
        return []

    def _gen_packages_items():
        for info in _parse_require_file(fname):
            parts = [info['package']]
            if with_version and 'version' in info:
                parts.extend(info['version'])
            if not sys.version.startswith('3.4'):
                # Package deps are broken in Python 3.4
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    parts.append(';' + platform_deps)
            yield ''.join(parts)

    return list(_gen_packages_items())


def _get_install_mode():
    """Determine installation mode (symlink or copy)."""
    if 'develop' in sys.argv:
        # Installed by `pip install -e .`
        return 'copy' if platform.system() == 'Windows' else 'symlink'
    elif any(cmd in sys.argv for cmd in ['sdist', 'bdist_wheel']) or platform.system() == 'Windows':
        # Installed by `pip install .` or creating source distribution
        # Use copy mode on Windows since symlink fails
        return 'copy'
    return None


def _create_symlink_or_copy(src_path, tar_path, mode):
    """Create symlink or copy file/directory."""
    if osp.exists(tar_path):
        if osp.isfile(tar_path) or osp.islink(tar_path):
            os.remove(tar_path)
        else:
            shutil.rmtree(tar_path)

    if mode == 'symlink':
        src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
        try:
            os.symlink(src_relpath, tar_path)
        except OSError:
            # Creating symlink may fail on Windows due to privileges
            # Fall back to copy mode
            mode = 'copy'
            warnings.warn(
                f'Failed to create symbolic link for {src_relpath}, '
                f'falling back to copy mode for {tar_path}'
            )

    if mode == 'copy':
        if osp.isfile(src_path):
            shutil.copyfile(src_path, tar_path)
        elif osp.isdir(src_path):
            shutil.copytree(src_path, tar_path)
        else:
            warnings.warn(f'Cannot copy file {src_path}.')
    else:
        raise ValueError(f'Invalid mode {mode}')


def add_mim_extension():
    """Add extra files required to support MIM into the package."""
    mode = _get_install_mode()
    if mode is None:
        return

    filenames = ['tools', 'configs', 'model-index.yml', 'dataset-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmseg', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        if osp.exists(src_path):
            tar_path = osp.join(mim_path, filename)
            _create_symlink_or_copy(src_path, tar_path, mode)


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='mmsegmentation',
        version=get_version(),
        description='Open MMLab Semantic Segmentation Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='MMSegmentation Contributors',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, semantic segmentation',
        url='https://github.com/open-mmlab/mmsegmentation',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'mim': parse_requirements('requirements/mminstall.txt'),
            'multimodal': parse_requirements('requirements/multimodal.txt'),
        },
        ext_modules=[],
        zip_safe=False
    )