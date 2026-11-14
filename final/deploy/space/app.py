"""Punto de entrada del Space de Hugging Face (ZeroGPU) de MacGyver Gym Rat.

Este archivo va en la RAÍZ del repo del Space, junto a `README.md` y
`requirements.txt` de esta misma carpeta. No copia el proyecto: al arrancar
clona el repo de GitHub en la versión `REPO_REF` y corre `final/deploy/app.py`.
Así el Space sirve exactamente el código del repo, y el catálogo con sus GIF
llega por el submódulo del dataset (los repos de Spaces no siguen submódulos).

`REPO_REF` es una variable del Space (Settings > Variables): una rama, un tag o
un commit. Para la entrega, fijarla al commit que se presenta.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

# ZeroGPU pide importar `spaces` antes que torch.
import spaces  # noqa: F401

REPO = "https://github.com/iamcroody/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros.git"
REF = os.environ.get("REPO_REF", "main")
CHECKOUT = Path(__file__).resolve().parent / "repo"


def git(*args):
    subprocess.run(["git", *args], check=True)


if not (CHECKOUT / ".git").exists():
    git("clone", REPO, str(CHECKOUT))
    git("-C", str(CHECKOUT), "checkout", REF)
    git("-C", str(CHECKOUT), "submodule", "update", "--init", "--recursive", "--depth", "1")

# Se carga por ruta con otro nombre: este archivo también se llama app.py.
spec = importlib.util.spec_from_file_location(
    "macgyver_app", CHECKOUT / "final" / "deploy" / "app.py")
macgyver_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(macgyver_app)
macgyver_app.main()
