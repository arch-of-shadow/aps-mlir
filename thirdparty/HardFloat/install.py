#!/usr/bin/env python3
"""Install the self-contained HardFloat package into a CIRCT module library."""
import argparse
import os
from pathlib import Path
import shutil

import yaml

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
WRAPPERS = ('HF32AddSub', 'HF32Mul', 'HF32Binary', 'HFCompute',
            'HFNativePipeline', 'HFUnit')


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        shutil.copyfile(source, destination)


def install(circt_root):
    library = circt_root / 'lib/Dialect/Cmt2/ModuleLibrary'
    if not (library / 'manifest.yaml').is_file():
        raise ValueError(f'CIRCT module library not found: {library}')
    installed_library = circt_root / 'build/share/circt/ModuleLibrary'
    destination = installed_library / 'firrtl/hardfloat_ip/package'
    for name in ['build-hardfloat-ip.py', 'ip_catalog.yaml', 'SOURCE.sha256',
                 'COPYING.txt', *[name + '.sv' for name in WRAPPERS]]:
        source = PACKAGE / name
        if not source.is_file():
            raise ValueError(f'Missing HardFloat package file: {source}')
        copy_file(source, destination / name)
    for source in (PACKAGE / 'source').rglob('*'):
        if source.is_file():
            copy_file(source, destination / source.relative_to(PACKAGE))
    copy_file(ROOT / 'examples/resource_ihp130.json', destination / 'resource-base.json')
    copy_file(PACKAGE / 'module-build.sh', destination.parent / 'build.sh')
    (destination.parent / 'build.sh').chmod(0o755)
    manifest = yaml.safe_load((library / 'manifest.yaml').read_text())
    definitions = yaml.safe_load((PACKAGE / 'cmt2-modules.yaml').read_text())['modules']
    names = {module['name'] for module in definitions}
    modules = [module for module in manifest['modules'] if module['name'] not in names]
    for module in modules:
        module['path'] = os.path.relpath(library / module['path'], installed_library)
    manifest['modules'] = modules + definitions
    target_manifest = installed_library / 'manifest.yaml'
    text = yaml.safe_dump(manifest, sort_keys=False)
    if not target_manifest.exists() or target_manifest.read_text() != text:
        target_manifest.write_text(text)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circt-root', type=Path, default=ROOT / 'circt')
    args = parser.parse_args()
    try:
        print(install(args.circt_root.resolve()))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
