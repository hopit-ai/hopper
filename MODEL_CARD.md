---
license: other
license_name: research-and-demo
license_link: https://github.com/hopit-ai/hopper/blob/main/MODEL_CARD.md
base_model: Qwen/Qwen3.5-4B
library_name: peft
language:
- en
tags:
- lora
- peft
- base_model:adapter:Qwen/Qwen3.5-4B
- qwen3.5
- decision-making
- calibration
- jevbench
datasets:
- allenai/ai2_arc
- tau/commonsense_qa
- cais/mmlu
- stanfordnlp/snli
- nyu-mll/multi_nli
- tals/vitaminc
- google/boolq
- rajpurkar/squad_v2
- clinc/clinc_oos
- fancyzhx/dbpedia_14
- nvidia/HelpSteer2
---

# Hopper

> **Research and demo use only.** This adapter is published for research and demonstration. Its training data included passages from RACE (via the `cais/mmlu` auxiliary set), which its authors release for non-commercial research only and whose terms extend to derived data. Do not use this adapter commercially. A version trained without these passages is in development.

Hopper is a LoRA adapter (rank 16) for
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It is built for the
[JevBench](https://github.com/fstandhartinger/jevbench) setting: a document, a policy and a
question go in, and a probability distribution over a fixed set of options comes out.

- **One forward pass per decision**, with thinking off. No text is generated. The answer is a
  softmax over the logits of the option letters (A, B, C, ...), restricted to as many letters as
  there are options.
- **A calibration map** (`hopper.json`) rescales that distribution by one temperature per answer
  type: choice 0.790, noul 0.753, score 0.900. It was fitted only on our own held-out
  JevBench-style items, never on a JevBench item. The map never changes the top answer — it
  divides log-probabilities by a positive number, which cannot reorder them.
  (In 1.0.0 the map was instead a bounded linear function of option count, state length,
  JSON-or-not, answer type and the entropy of the model's own distribution. A leave-one-source-out
  ablation showed that map was worth +0.1 Calibration over no map at all on sources its fitting
  set had never seen, against +4.5 and +2.9 for the per-answer-type map fitted on the same data,
  so 1.1.0 replaced it. Answers are identical either way; only the confidences move. The old map
  ships alongside as `hopper-v1.0-linear.json`.)
- **Serving code**: [github.com/hopit-ai/hopper](https://github.com/hopit-ai/hopper). It runs an
  HTTP server with the JevBench `/v1/systemone` wire format. At load it merges the adapter into the
  bf16 weights, and it refuses to start if the fast linear-attention kernels are not active.

The serving code is licensed Apache-2.0. The adapter weights are offered for research and demo use
only, because of the RACE training-data terms described above and under "Training data". The base
model is Apache-2.0 ([licence](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)), and this
adapter does not change its terms.

## Intended use

Hopper makes single-step policy decisions over a short document: yes/no (`noul`), choice among
named options, and ordinal scores. It returns calibrated probabilities, and it is meant to be run
and measured on JevBench. It is not a chat model. It is also not meant for decisions with legal,
medical, financial or safety consequences unless a person reviews them.

## Prompt format

The chat template of Qwen3.5-4B is applied with `enable_thinking=False` and a generation prompt.

- **System**: `You make decisions about a document under a policy. Read only what is written in the document. Reply with the letter of the correct option and nothing else.`
- **User**: one JSON object,
  `{"evidence": <document>, "criterion": <policy>\n\n<question>, "options": [{"letter": "A", "description": "<label>: <description>"}, ...]}`.
  A yes/no question has the options `true` and `false`, and its rubric is appended to the policy.
  The policy for JevBench items is `Decide the case using only what the document states. Exactly one option is correct.`

The readout is the next-token logits at the end of the prompt, restricted to the option letters,
then a softmax, then the calibration map. `hopper_decisions/request.py` and `prompt.py` in the code
repository build this prompt exactly.

## How to load it

The easiest way to serve it is with the package (`pip install` the code repository, see its README):

```python
from hopper_decisions import Decider
decider = Decider(adapter="HopitAI/hopper")      # the packaged calibration map is the default
decider.decide({"state": "The customer wants a refund for order 12.",
                "questions": {"decision": {"type": "choice", "instructions": "Route the ticket.",
                                           "criteria": {"refund": "money back", "track": "where is it"}}}})
```

Or with transformers and peft directly:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, revision = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "HopitAI/hopper").merge_and_unload().eval()

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_json}]   # as above
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
letters = [tokenizer.encode(l, add_special_tokens=False)[0] for l in "AB"]        # one letter per option
with torch.inference_mode():
    logits = model(input_ids=ids.to("cuda")).logits[0, -1, letters].float()
probs = torch.softmax(logits, -1)   # before the calibration map
```

Qwen3.5's linear-attention layers need `flash-linear-attention==0.5.2` and `causal-conv1d` 1.7.0
to run at full speed. Without them, transformers silently falls back to a path that is more than
10x slower. The pinned versions are `torch==2.8.0`, `transformers==5.17.0`, `peft==0.21.0` and
`accelerate==1.15.0`.

## Training data

The adapter was trained on a mix of three sources:

1. **Synthetic decision families made by LLM-based generation.** An LLM wrote the families, and
   their labels were computed in code.
2. **A JevBench-style set, also made by LLM-based generation.** Items were kept only where
   independent LLM solvers agreed with the answer. It contains no JevBench item. Every item was
   checked against all public JevBench questions and states (normalised question identity, and
   any shared 8-word sequence) and dropped on a match.
3. **Public human-labelled datasets.** We used examples from their training splits, converted
   into the decision format above. Each is used under its own licence:

| dataset | used for | licence |
| --- | --- | --- |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge, ARC-Easy) | multiple choice | CC BY-SA 4.0 |
| [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | multiple choice | MIT |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (`auxiliary_train`) | multiple choice | MIT (as stated on the dataset card), but the auxiliary set bundles other public datasets, including RACE, whose authors release it for non-commercial research only, with terms that extend to derived data |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | entailment | CC BY-SA 4.0 |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | entailment | CC BY 3.0 / CC BY-SA 3.0 / MIT / other, per source genre (see the dataset card) |
| [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc) | fact verification | CC BY-SA 3.0 |
| [google/boolq](https://huggingface.co/datasets/google/boolq) | yes/no questions | CC BY-SA 3.0 |
| [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | answerability | CC BY-SA 4.0 |
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) | intent classification | CC BY 3.0 |
| [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | topic classification | CC BY-SA 3.0 |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | response-quality judgement | CC BY 4.0 |

The calibration map was fitted only on our own held-out JevBench-style items — the same fitting
set in 1.1.0 as in 1.0.0, with the 66 leakage-flagged items dropped. It never saw a JevBench item.

## Evaluation

**These are local numbers, not official JevBench results.** They were computed with our own
evaluation path on the 231 public JevBench items (argmax over the exact label set, with ties
going to the smallest label as in `jevbench/scoring.py`). The judge tier and the held-out items
are not public, so they are not included. Only the JevBench maintainer's run on his own GPU is
official.

| tier | items | Hopper | same base, frozen, same prompt and map type |
| --- | ---: | ---: | ---: |
| easy | 48 | 1.000 | 1.000 |
| standard (original) | 72 | 0.944 | 0.958 |
| hard | 111 | 0.685 | 0.631 |

The accuracies above are the same in 1.1.0 as in 1.0.0: no calibration map can move an answer.

On the hard tier, top-label ECE is 0.102, and distribution fidelity (1 − mean total-variation
distance) on the 10 public probability items is 0.830. **Both were measured with the 1.0.0
calibration map**, and we have not recomputed them for 1.1.0, because the reserved half of the
public items may be scored only under the pre-registered rule and we will not spend it again on
a map change. On the development half alone, computed on our saved predictions, replacing the
1.0.0 map with the 1.1.0 one takes hard-tier ECE from 0.159 to 0.112 and the harness's Calibration
axis from 76.5 to 79.4, with the top answers unchanged. Treat that as an estimate and not a
promise: re-running the same 55 hard items through the server applies the identical map and lands
at 76.0, because a binned ECE over 55 items is not stable to the bf16 differences between two runs
of the same weights. The case for the map rests on a 517-item held-out set, where it is worth +4.5,
not on this fold.

**Disclosure.**
- The public items were split in half before we started. The half we developed on (115 items)
  was used as a development gate many times: 26 distinct model and prompt configurations, plus
  more than twenty calibration-map variants. Our JevBench-style training set's style sheet was
  written by reading that half, and some of its training items target behaviours we saw fail on
  its hard items. On that half the adapter scores hard 0.709.
- The other half (116) was kept as a reserve and scored only in aggregate. Our models were
  predicted on it in three earlier sessions (other configurations) and once for this system,
  chosen beforehand by a pre-registered rule. On it the adapter scores hard 0.661 (37 of 56) and
  the frozen base 0.643 (36 of 56): it is level with the frozen model on accuracy there, not ahead.
- The calibration map was fitted only on our own held-out JevBench-style items, never on a
  JevBench item. The 1.1.0 map was fitted on exactly the same items as the 1.0.0 map; it was
  chosen over it on out-of-pool folds of our own data, not on any JevBench score, and the
  reserved half was not re-run for it.
- No JevBench item or paraphrase was used in training. Every item we wrote was checked against
  all public JevBench questions and states (normalised question identity, and any shared 8-word
  sequence) and dropped on a match; the check reads only hashes and reports only counts.
- Expect the held-out hard items to score below the public ones, and expect the judge tier, which
  we have never seen, to be the least predictable part.

## Limitations

- **One pass of a 4B model.** Hopper does not reason step by step. Any problem that needs a chain
  of intermediate results is decided in one forward pass by Qwen3.5-4B.
- **Dates and multi-step arithmetic are weak.** Date differences, deadlines and chained
  calculations fail often.
- **Long documents that need several hops are weak.** Accuracy drops when the answer needs facts
  from several distant parts of a long document.
- **The calibration was fitted on our own data.** The map was fitted on our own JevBench-style
  items. On a different distribution of questions, its confidences can be off. 1.1.0 made the map
  as simple as we could justify — three temperatures instead of seven coefficients — precisely
  because the richer map did not transfer off the pool it was fitted on.
- **The dev half flatters it.** On the reserved half of the public items, it is level with the
  frozen base model on hard-tier accuracy (see the disclosure).
- **Tested only on English.** We have not measured any other language.
