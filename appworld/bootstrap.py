"""Install dependencies and unpack bundled official data without a data download."""
import os
from pathlib import Path
import subprocess
import sys
import venv
import platform

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('This self-contained release requires Python 3.12.')
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise SystemExit('This self-contained release is validated on Linux x86_64 only.')
    os.chdir(ROOT)
    target = ROOT / '.venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    command = [str(python), '-m', 'pip', 'install']
    wheels = ROOT / 'assets/wheels-linux-py312'
    if sys.platform == 'linux' and sys.version_info[:2] == (3, 12) and platform.machine() == 'x86_64' and wheels.exists():
        command += ['--no-index', '--find-links', str(wheels), '-c', 'requirements-linux-py312.txt']
        print('Using bundled Linux x86_64 / Python 3.12 dependency wheels.', flush=True)
    else:
        raise SystemExit('Bundled Linux/Python 3.12 wheels are missing; refusing external downloads.')
    subprocess.run(command + ['-r', 'requirements.txt'], check=True)
    subprocess.run([str(python), 'scripts/prepare_data.py'], check=True)
    print(f'Ready. Run: {python} run.py --help')


if __name__ == '__main__':
    main()
