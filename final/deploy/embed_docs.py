"""Calcula los embeddings de los documentos del catálogo en un proceso aparte.

    python final/deploy/embed_docs.py <salida.npy>

Solo lo usa la app en un Space de ZeroGPU. Allí, codificar los 1324 documentos
con sentence-transformers dentro del proceso principal deja a los workers de
ZeroGPU sin GPU: cada petición falla en `worker_init` con "No CUDA GPUs are
available". Lo medimos en el Space: con el generador, los documentos y el índice
cargados al arrancar, una función que solo hace `torch.zeros(1).cuda()` falla;
con el generador solo, funciona.

Este script hace la misma codificación que `m3/rag.py::DenseIndex` (mismo
modelo, mismos documentos y en el mismo orden, `batch_size=64`,
`normalize_embeddings=True`) en un proceso que no comparte estado con la app, y
guarda la matriz. La app la pone en `cuda` (`app.GpuDenseIndex`) sin ejecutar
el modelo.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "eval", "m3", "final"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def main(out_path):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    import corpus as C
    from efficiency import efficiency as E

    docs = C.build_documents()
    encoder = SentenceTransformer(E.EMBED_MODEL, device="cpu")
    embeddings = encoder.encode([d["text"] for d in docs], batch_size=64,
                                normalize_embeddings=True, show_progress_bar=False)
    np.save(out_path, embeddings)
    print(f"[embed_docs] {len(docs)} documentos -> {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
