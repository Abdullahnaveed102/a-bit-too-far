# A Bit Too Far

Supporting code for **A Bit Too Far: Understanding Large Language Model Resilience under Increasing Fault Intensity**.

Abdullah Naveed and Guanpeng Li, University of Florida.

We study how fault intensity, the distribution of corrupted bits, and the bit operation affect LLM inference. This repository provides the prompt construction, answer scorer, activation hooks, and settings for the controlled experiments in the paper.

## Contents

- [`src/fi_campaign/`](src/fi_campaign/): scoring, prompt construction, native bit operations, and projection-output hooks.
- [`configs/`](configs/): model revisions, generation settings, and experiment grids.
- [`docs/experiments.md`](docs/experiments.md): experiment design, calibration, measurement, and dataset sources.
- [`examples/`](examples/): small examples using synthetic inputs.
- [`tests/`](tests/): scoring, prompting, mutation, and hook tests.

Datasets, generated responses, model weights, calibration profiles, and per-question fault plans are not distributed here. The code is a companion to the paper, not a complete campaign launcher or a release of the original evaluation cohorts.

## Quick start

Python 3.10 or newer is sufficient for scoring and bit operations. From the repository root:

```bash
PYTHONPATH=src python examples/score_response.py --task gsm8k --gold 7 --response "Final answer: 7"
PYTHONPATH=src python examples/score_response.py --task squad2 --unanswerable --response "Final answer: UNANSWERABLE"
PYTHONPATH=src python examples/bit_operations.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The examples run without models or datasets. Tests that require PyTorch are skipped when it is not installed. To import the package from another project, install it with `python -m pip install -e .`.

## Scoring and prompts

[`scoring.py`](src/fi_campaign/scoring.py) contains the scorer used in the controlled experiments (`final_answer_v1.4`). It reads the text on the same line after the first `Final answer:` or `####` marker. Later marked answers do not replace it. Without a marker, the full response is evaluated.

```python
from fi_campaign.scoring import score_output

result = score_output("Final answer: 7", ["7"], "gsm8k")
print(result["valid"], result["exact"])
```

An unresolved extraction has `valid=False` and `exact=None`. It remains in the denominator when computing correctness over all assigned executions, but receives no correctness credit. This scorer is separate from the answer-presence metric used in the broad campaign.

[`prompting.py`](src/fi_campaign/prompting.py) contains the task templates and synthetic demonstrations. Instruction-tuned models use their pinned native chat templates; Llama-2 uses the plain few-shot template. The model and template settings are in [`controlled_protocol.json`](configs/controlled_protocol.json).

To call `render_task_prompt(model, item)`, load a model entry from that configuration. The input item needs `task` (`gsm8k` or the internal QA name `squad`), a nonempty source `prompt`, and `question`; QA also needs `context`. The scorer accepts both `squad` and `squad2`. `render_task_prompt` returns the task text before chat wrapping; `freeze_final_prompt` additionally needs the pinned tokenizer for chat models.

## Fault injection

[`hooks.py`](src/fi_campaign/hooks.py) implements transient mutations at decoder projection outputs and the nonfinite-only, finite-only, and combined repair policies. [`words.py`](src/fi_campaign/words.py) selects and modifies native 16-bit positions. Forcing a bit to its existing value counts as selected but not changed.

The hook code requires PyTorch. Native chat rendering also requires the matching tokenizer. Optional dependencies are listed in [`requirements-models.txt`](requirements-models.txt); the recorded GPU environment is documented with the configurations. Install a PyTorch build appropriate for your hardware.

`ActivationHooks` is a low-level context manager. The caller supplies the model and fault plan, advances `hooks.step` for each generation forward pass, and supplies any calibration bounds. It does not load models, generate tokens, collect calibration profiles, or submit cluster jobs.

The scorer, prompt renderer, bit operations, and hooks are copied unchanged from the source snapshot used for the protocol and geometry experiments. Campaign launchers and dedicated timing wrappers are not included; timing the hooks alone does not reproduce the paper's runtime benchmark.

## Before publishing changes

```bash
python scripts/check_release.py
```

This checks tracked and unignored files for large files, excluded data formats, private paths, and common credential patterns. It does not replace a manual review of the files being committed.
