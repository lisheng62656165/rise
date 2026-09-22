# Bundled assets

- `data-0.1.0.bundle`: official encrypted APPWorld dataset, downloaded from
  https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle .
- `appworld-0.1.3.post1-py3-none-any.whl`: official PyPI wheel; includes the
  encrypted apps bundle. Do not replace with decrypted apps source.
- `wheels-linux-py312/`: dependency wheels for Linux x86_64 / CPython 3.12.
  Versions are recorded in `../requirements-linux-py312.txt`. Original license
  notices are retained inside each wheel's dist-info directory. Each dependency
  retains its own license, not the research-code license.

Keep these files when distributing the self-contained artifact. The wheels
target modern glibc Linux (2.28 or newer), not Alpine/musl or ARM. Other platforms
use automatic online dependency installation. Python/OS and API access are not
bundled. The extracted protected data is local-only; see ../NOTICE.md.
