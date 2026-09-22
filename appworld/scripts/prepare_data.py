"""Materialize protected assets locally; never publish the runtime directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['APPWORLD_ROOT'] = str(ROOT / 'runtime')


def main():
    from appworld.common.constants import PASSWORD, SALT
    from appworld.common.utils import unpack_bundle
    from appworld.install import install_package
    install_package()
    runtime = ROOT / 'runtime'
    runtime.mkdir(exist_ok=True)
    if not (runtime / 'data/datasets/test_challenge.txt').exists():
        unpack_bundle(bundle_file_path=str(ROOT / 'assets/data-0.1.0.bundle'),
                      base_directory=str(runtime), password=PASSWORD, salt=SALT)
    from appworld import load_task_ids
    for name, expected in [('test_normal', 168), ('test_challenge', 417)]:
        actual = len(load_task_ids(name))
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}')
        print(f'{name}: {actual} tasks')


if __name__ == '__main__':
    main()
