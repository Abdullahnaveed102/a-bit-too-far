# Experimental setup

The paper combines broad characterization with matched geometry and protection
experiments. Their prompts and answer metrics differ, so results are analyzed
separately. The JSON files in `configs/` record the experimental settings;
campaign launchers are not included.

## Models, inputs, and generation

The controlled studies use three models on GSM8K and SQuAD2, giving six
model/task configurations:

| Model | Model repository |
| --- | --- |
| Llama-2-7B, base | `meta-llama/Llama-2-7b-hf` |
| Mistral-7B, instruction tuned | `mistralai/Mistral-7B-Instruct-v0.3` |
| Granite-3.3-8B, instruction tuned | `ibm-granite/granite-3.3-8b-instruct` |

[controlled_protocol.json](../configs/controlled_protocol.json) pins their model
and tokenizer revisions and records the software versions. Obtain model files
from the listed repositories under their respective access terms. Generation
uses evaluation mode, FP16, eager attention, greedy token selection, one example
per request, and a 256-new-token limit. Natural generation stops at the first EOS;
composition instead uses a fixed 256-token horizon. Scoring uses the generated
prefix bounded by the first EOS. The recorded GPUs were NVIDIA B200s; CUDA 12.8
is the PyTorch build version, not a separately recorded runtime version.

Llama-2 uses plain prompts with two synthetic task demonstrations. Mistral and
Granite use their pinned native chat templates. The exact task strings are in
[prompting.py](../src/fi_campaign/prompting.py). Granite receives the fixed system
message in the configuration to avoid a date-dependent template default.

Download [GSM8K](https://github.com/openai/grade-school-math) and
[SQuAD2](https://rajpurkar.github.io/SQuAD-explorer/) from their original sources.
Calibration uses training inputs; evaluation uses GSM8K test and SQuAD2
development inputs. Question/context units are deduplicated, compatible answer
references are merged, and conflicting labels are excluded before sampling.
Each controlled cohort has 200 questions per task, shared across the three
models. These are sampled cohorts, not full-benchmark evaluations. Dataset
exports, selected question lists, generated outputs, and calibrated profile
values are not included here; the settings alone do not reconstruct the exact
evaluation cohorts.

## Controlled studies

| Study | Main design | Assigned executions |
| --- | --- | ---: |
| Protocol controls | Natural/fixed stopping; no-hook, empty-hook, identity write, and 1-/16-bit flip arms | 24,000 |
| Equal-budget geometry | Three operations; 16/32/64 selected bits; 1/2/4/8/16 bits per value | 108,000 |
| Fault composition | Three placements; three operations; no fault, A, B, and exact union AB | 86,400 |
| Initial calibration assessment | Sampled-quantile policies; healthy arms and 4-/16-bit faults | 69,600 |
| Matched calibration comparison | Min/max versus sampled-quantile healthy arms; matched min/max fault-policy replays | 28,800 |
| Protection across intensity | Bit flip/stuck-at-one; 4/16/64 selected bits; 1/4 bits per value; four policies | 57,600 |
| Healthy-input runtime | Five arms; 30 questions/task; three timed repetitions | 2,700 |

Counts include the listed healthy arms and policy replays, but exclude standalone
references, calibration, warmups, and acceptance checks. The protection-intensity
grid includes 9,600 reused executions from the matched calibration comparison;
those are not additional replications.

### Fault geometry and composition

[fault_geometry.json](../configs/fault_geometry.json) specifies these designs.
Forward hooks mutate the native outputs of decoder q/k/v/o and gate/up/down
projections, excluding the language-model output head. Bit flip toggles selected
positions; stuck-at-zero and stuck-at-one force them once. A selected bit need
not change if it already has the forced value.

Equal-budget comparisons share the question, draw, and uniformly selected target
projection. Coordinates are sampled for each geometry, without replacement
within a tensor; bit positions are sampled without replacement within a value.
Coordinates across geometries need not be identical or nested. Injections occur
at prefill. Composition compares disjoint 8-bit components with their exact
16-bit union, using two sites per component and four bits per site. Placements
span one tensor, different layers, or successive decode invocations. Each
question has two draws in these studies.

### Calibration and protection

[protection.json](../configs/protection.json) gives the grids and policy rules.
Profiles are computed per module from 128 clean prefills. Min/max profiles
observe all values. Sampled quantiles use 128 evenly spaced values per
module/input and a reservoir of at most 8,192 values, with bounds at 0.1% and
99.9%. The matched comparison collects both profiles during the same forward passes.
Bounds are expanded by 1.25 about their midpoint and capped at FP16 finite
limits.

Nonfinite repair maps NaN to zero and signed infinities to the corresponding
bounds. Finite clipping clamps finite values only; combined protection does both.
Repair follows injection in the same hook and runs at every covered projection
invocation, including healthy ones. Policies replay the same selected fault plan;
earlier protection can change a later mutation's input value.

Geometry, protection-intensity evaluation, and healthy policy selection use
separate cohorts and clean references. Min/max bounds are selected using a
zero-observed-answer-loss criterion on the healthy policy-selection cohort and
frozen before fault evaluation. The runtime study reuses 30 questions per task
from that selection cohort; it is not an independent healthy holdout. It uses
interleaved arms, three repetitions, and synchronized generation and outer-service
timers. Loading, warmups, references, calibration, and file I/O are excluded;
changes in generated length remain in the measured cost.

## Measurements and pairing

Controlled correctness uses the fixed `final_answer_v1.4` extraction rules. The
first `Final answer:` or `####` marker selects the text after it on the same line;
without a marker, the scorer uses the full response. GSM8K requires an
unambiguous normalized number. QA uses normalized exact matching with explicit
no-answer handling and a 32-word/240-character extraction limit. Empty marked
answers, ambiguous numbers, and invalid formats are unresolved. The released
scorer defines the full rules, including rejection cases.

Correctness uses all assigned executions. Clean-correct retention conditions on
a scorer-correct clean reference. Rescues and harms are paired transitions from
noncorrect to correct and from correct to noncorrect, respectively. Unresolved
extraction remains separate from a valid wrong answer. Token-F1 is a lexical
sensitivity check, not independent semantic judgment.

Aggregate controlled estimates equally weight the six configurations. Uncertainty
is obtained by task-stratified question/context-cluster bootstrap, keeping paired
models, draws, arms, and repetitions together. Intervals are pointwise 95%
intervals. Runtime averages paired log-latency ratios over repetitions within a
question before resampling; exponentiating gives the geometric-mean ratio.

## Broad characterization

The broad campaign comprises 1.62 million injected executions across 12 models
and 18 model/dataset configurations, using 500-prompt pools and a 100-new-token
limit. Its four fault patterns are isolated
single-bit faults, 2/4/8/16 bits in one value, one bit across up to 60 sites, and
compound faults with 1--20 sites and 1--16 selected bits per site. Unlike the
controlled study, projection sampling is weighted by the input/output dimension
product. Compound exposure is the sum of selected bits over delivered sites.

Broad mismatch compares stripped generated strings. Reference-answer loss checks
answer presence among clean answer-positive responses, including answers within
intermediate reasoning; it is not the controlled final-answer correctness metric.
The broad study also uses a historical XTREME QA export. Its precise subtask and
split are not established by the retained export metadata, and controlled model
revision pins should not be applied to that historical campaign.
