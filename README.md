# SA + MTP Hybrid Speculator (b12x / vLLM)

Lossless **suffix-automaton speculative decoding** layered on top of a model's native **MTP** draft head, for the [b12x](https://github.com/lukealonso/b12x) vLLM runtime. It ports Baseten's [`sa_spec`](https://github.com/basetenlabs/sa_spec) engine — which they ship for TensorRT-LLM — into vLLM's V2 MTP speculator.

On output that **overlaps the context** (agentic file edits, regeneration, RAG, structured/JSON, repetitive code) it reaches accept-lengths of ~10–16 and **~2.7× single-stream speedups** (measured 69 → 182 t/s). The honest tradeoff: it costs **~15% on novel generation** — SA needs a wide draft to copy long runs, and that wide draft is verified every step. Always lossless. **This is a real tradeoff, not free — read [Measured performance](#measured-performance) before you ship it.**

## What it does

Two drafters, one switch:

- **MTP** (the model's multi-token-prediction head) drafts the novel tokens — but its acceptance dies past ~5 positions, so we **cap** it there (`MTP_DRAFT_DEPTH`).
- A **suffix automaton** indexes the prompt + everything generated so far. On any decode step where the current output's suffix matches earlier context by **≥ `SA_SPEC_THRESH`** tokens, it overlays a long **copied** draft at the full speculative width.
- The rejection sampler verifies whatever draft ids land in the batch, so it's **lossless by construction** (greedy: accept iff draft == target argmax) — copied tokens are verified exactly like neural drafts. No verify-path changes.

Result: accept-len ~10–16 wherever the model is reproducing context, at a **~15% cost on novel generation** (decomposed below). Background: Baseten's [writeup](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/) and the [SAM-Decoding paper](https://arxiv.org/abs/2411.10666).

## Measured performance

Rigorously benchmarked on **GLM-5.2-NVFP4** (RTX PRO 6000 ×4, TP4, `FLASHINFER_MLA_SPARSE_SM120`, DCP4), single-stream, temp 0, **4-run averages** (`measure_ab.py`):

| workload | base (no SA) | + SA-MTP | delta |
|---|---|---|---|
| novel codegen | 56 t/s | 48 t/s | **−14%** |
| novel prose | 37 t/s | 32 t/s | **−14%** |
| **reproduce / edit existing** | 69 t/s | **182 t/s** | **+2.6×** |

**It's a ~15% generation tax for a ~2.7× copy/edit win.** Ship it if your day is edit/reproduce-heavy (agentic file rewriting, RAG, regeneration, long structured output); skip it for from-scratch generation.

### Why the tax exists (decomposed, measured)

- **~5% is physics.** SA needs a **wide** draft (`num_speculative_tokens=16`) to copy long runs, and that same wide draft is verified on *every* step — including novel ones where SA sits out. One draft tensor can't be wide-for-copy and narrow-for-novel at the same time.
- **~10% is software.** The CPU suffix-automaton syncs once per step inside `sa_spec.extend`. Recoverable only by a **GPU-resident automaton** — a real rewrite, not a config. (Lifting it would reach ~53 t/s codegen, the true ceiling for SA-with-spikes.)

### What does NOT work (so you don't repeat it)

- **Lean-on-miss / variable draft width** (narrow verify on novel, wide on copy) to dodge the verify tax: the narrow verify isn't a captured cudagraph size, so it runs **eager** and is *slower* than the wide captured verify (measured 22–36 t/s everywhere). The wide captured verify is the fastest available.
- **Disabling async scheduling** (so a narrow draft count reaches the scheduler): costs ~7% on its own and the lean still loses.
- True taxless-with-spikes needs **both** capturing two verify cudagraph widths **and** a GPU automaton — two deep rewrites.

### Lossless

Greedy rejection sampling accepts a copied token only if it equals the target's argmax, so output is identical to no-spec greedy. Verified **byte-identical** on deterministic-repetition prompts (`measure_det.py`). Note: this stack is non-deterministic on near-ties at temp 0 — like the base model itself — so a whole-output hash won't match run-to-run; that's the base's property, not a speculation error, which is why the lossless check uses forced-deterministic prompts.

## Requirements

- A **b12x vLLM image** serving a model that uses the `mtp` speculative method on the V2 model runner (GLM-5.2, DeepSeek-V3/R1, …).
- Ability to compile `sa_spec` for your GPU arch (`CUDA_ARCH`; `120` = Blackwell / RTX PRO 6000, `90` = Hopper, `100` = B200).
- Serve at **temperature 0** with **greedy** drafting for the lossless copy benefit.

## Build

```bash
docker build --build-arg BASE=<your-b12x-vllm-image> \
             --build-arg CUDA_ARCH=120 -t vllm-sa-mtp .
```

Builds `sa_spec` for your arch, installs the `sa_mtp` speculator, and applies the patch (idempotent). For batch > 32 or context > 256k, pass e.g. `--build-arg SA_CMAKE_EXTRA="-DC_MAX_SLOTS=64"`.

> **Known-good base** (GLM-5.2-NVFP4 on sm120 / RTX PRO 6000), the build these anchors are cut against:
> `voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626` — the measured numbers above are from this image + [PR#56 (FlashInfer-sm120 DCP)](https://github.com/local-inference-lab/vllm/pull/56) at DCP4.
> Any other b12x build may shift the patch anchors — the patcher will tell you (`anchor not found`).

## Use

Set on the vLLM container:

| env | what |
|---|---|
| `SA_MTP=1` | **enable the hybrid.** Omit it → stock MTP (the SA code isn't even loaded — clean A/B). |
| `SA_SPEC_THRESH` | min context-match length before SA copies (`sa_spec` default 8; we run **4**). Lower = SA fires more. `inf` = SA muted. |
| `MTP_DRAFT_DEPTH` | cap on MTP draft forwards/step (default **5**; **3** matches the base draft width). Cuts wasted draft forwards, but the verify stays full-width — it does **not** remove the novel tax (see [Measured performance](#measured-performance)). |

Your speculative config must be `method: "mtp"` with a **wide** `num_speculative_tokens` (e.g. **16** — the SA copy width) and `draft_sample_method: "greedy"`. See `docker-compose.example.yml`.

**To disable / get a baseline:** unset `SA_MTP` (full stock MTP), or keep it on with `SA_SPEC_THRESH=inf` (speculator loaded, SA muted).

## How it's wired (and the honest caveats)

This is a **monkey-patch**, not a clean upstream method. `apply_sa_patches.py` edits three anchor sites in the installed vLLM and drops in the `sa_mtp` package:

1. the V2 speculator **dispatch** (`spec_decode/__init__.py`) — env-gates `mtp` → `SAMTPSpeculator` when `SA_MTP=1`;
2. two **model-runner hooks** (`model_runner.py`) — `sa_add_request` / `sa_remove_request`, duck-typed (no-ops for any other speculator).

It deliberately keeps the method string `"mtp"`, so vLLM's MTP machinery and config validation stay untouched — **but that means the anchors are tied to a b12x build.** The patcher **fails loudly** if an anchor isn't found, so **pin your `BASE` image**. A small diagnostic logs SA match-depth every 96 steps (one GPU sync). See `TECHNICAL.md` for the exact integration points if you're porting to another b12x build.

## Credits / License

Ported by [MadeBy561](https://github.com/MadeBy561). Apache-2.0 (see `LICENSE`). Suffix-automaton engine: Baseten [`sa_spec`](https://github.com/basetenlabs/sa_spec) (Apache-2.0). Built on [b12x](https://github.com/lukealonso/b12x) / [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0). See `NOTICE`.
