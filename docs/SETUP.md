# Setup and run

[← Back to the README](../README.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git

## Setup

Clone the repo together with the dataset submodule:

```bash
git clone --recurse-submodules https://github.com/iamcroody/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros.git
cd JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros
```

If you already cloned it without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Install uv, if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies:

```bash
uv sync
```

## Run

`uv run` checks the lockfile and syncs the environment on every call, so there is no virtualenv
to activate:

```bash
uv run python scripts/01_macgyver_data.py     # builds data/processed/macgyver/{train,val,test}.jsonl
uv run python scripts/02_score.py             # self-test: scorer against gold answers
uv run python scripts/03_eval_zeroshot.py     # zero-shot baseline
uv run python scripts/04_train_lora.py        # trains models/r16-all-linear
uv run python scripts/03_eval_zeroshot.py --adapter models/r16-all-linear --report-name finetuned
```

M2, the evaluation harness:

```bash
uv run python eval/harness.py            # self-test: no model weights needed
uv run python eval/build_eval_set.py     # rebuilds eval/eval_set.json from the catalog
```

The full scorecard needs a GPU and runs in
[`notebooks/M2_harness.ipynb`](../notebooks/M2_harness.ipynb) on a free Colab T4.

M3, the agentic RAG:

```bash
uv run python m3/selftest.py             # self-test: no model weights needed
```

The index, the four systems, the scorecard and RAGAS need a GPU and run in
[`notebooks/M3_rag.ipynb`](../notebooks/M3_rag.ipynb) on a free Colab T4, which installs
the exact versions in `m3/requirements.txt`.

M5, the final integrator (efficiency + deploy):

```bash
uv run python final/efficiency/selftest.py   # self-test: efficiency plumbing (numpy only)
uv run python final/deploy/selftest.py        # self-test: deploy plumbing (no GPU/network)
uv run python final/ci/harness_smoke.py       # deterministic harness smoke (what CI runs)
```

The efficiency before/after table needs a GPU and runs in
[`notebooks/M5_efficiency.ipynb`](../notebooks/M5_efficiency.ipynb) on a free Colab T4; the Space
is deployed from [`final/deploy/`](../final/deploy/README.md). Both the M5 notebook and the Space
clone the current working branch / main.

Add a dependency with `uv add <package>`, it updates `pyproject.toml` and `uv.lock` for you.

## Dataset

The dataset lives in `data/exercises-dataset` as a submodule pinned to a fixed commit, so every
run trains on exactly the same data. See [`docs/DATASET.md`](../docs/DATASET.md) for source, task,
how the household-object mapping was built, splits, language, licence, and known biases.

To move the pin to the latest dataset commit:

```bash
git -C data/exercises-dataset pull origin main
git add data/exercises-dataset
git commit -m "build(data): bump dataset pin"
```

Exercise media is © [Gym visual](https://gymvisual.com/) and is referenced, not redistributed
here, see the dataset repo's `NOTICE.md` before reusing it.
