from pathlib import Path
from subprocess import run

pkg_path = Path(__file__).parent
models_path = pkg_path / "models"

if not models_path.exists():
    download_sh = pkg_path.parent / "download_models.sh"
    run(["sh", str(download_sh)])
