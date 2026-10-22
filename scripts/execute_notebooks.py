"""Execute supported root notebooks with a fresh kernel, then render HTML."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
out = Path('artifacts/notebooks')
out.mkdir(parents=True, exist_ok=True)
for path in sorted(Path('.').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute()
    nbformat.write(nb, out / path.name)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (out / (path.stem + '.html')).write_text(body)
    print('Executed and rendered:', path.name)
