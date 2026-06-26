# SA + MTP Hybrid Speculator (b12x / vLLM)

Lossless **suffix-automaton speculative decoding** layered on top of a model's native **MTP** draft head, for the [b12x](https://github.com/lukealonso/b12x) vLLM runtime. It ports Baseten's [`sa_spec`](https://github.com/basetenlabs/sa_spec) engine — which they ship for TensorRT-LLM — into vLLM's V2 MTP speculator.

On output that **overlaps the context** (agentic file edits, regeneration, RAG, structured/JSON, repetitive code) it reaches accept-lengths of ~10–16 and large single-stream speedups; on novel text it falls back to plain MTP. Always lossless.

## What it does

Two drafters, one switch:

- **MTP** (the model's multi-token-prediction head) drafts the novel tokens — but its acceptance dies past ~5 positions, so we **cap** it there (`MTP_DRAFT_DEPTH`).
- A **suffix automaton** indexes the prompt + everything generated so far. On any decode step where the current output's suffix matches earlier context by **≥ `SA_SPEC_THRESH`** tokens, it overlays a long **copied** draft at the full speculative width.
- The rejection sampler verifies whatever draft ids land in the batch, so it's **lossless by construction** (greedy: accept iff draft == target argmax) — copied tokens are verified exactly like neural drafts. No verify-path changes.

Result: ~MTP speed on novel generation, and accept-len ~10–16 wherever the model is reproducing context. Background: Baseten's [writeup](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/) and the [SAM-Decoding paper](https://arxiv.org/abs/2411.10666).

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
> `voipmonitor/vllm:glm52-dark-devotion-pr31-pr15-w4a16scale-vllm79f154c-b12xaecc88f-cu132-20260622`.
> Any other b12x build may shift the patch anchors — the patcher will tell you (`anchor not found`).

## Use

Set on the vLLM container:

| env | what |
|---|---|
| `SA_MTP=1` | **enable the hybrid.** Omit it → stock MTP (the SA code isn't even loaded — clean A/B). |
| `SA_SPEC_THRESH` | min context-match length before SA copies (`sa_spec` default 8; we run **4**). Lower = SA fires more. `inf` = SA muted. |
| `MTP_DRAFT_DEPTH` | cap on MTP draft forwards/step (default **5**). Keeps SA-miss steps at baseline-MTP cost. |

Your speculative config must be `method: "mtp"` with a **wide** `num_speculative_tokens` (e.g. **16** — the SA copy width) and `draft_sample_method: "greedy"`. See `docker-compose.example.yml`.

**To disable / get a baseline:** unset `SA_MTP` (full stock MTP), or keep it on with `SA_SPEC_THRESH=inf` (speculator loaded, SA muted).

## How it's wired (and the honest caveats)

This is a **monkey-patch**, not a clean upstream method. `apply_sa_patches.py` edits three anchor sites in the installed vLLM and drops in the `sa_mtp` package:

1. the V2 speculator **dispatch** (`spec_decode/__init__.py`) — env-gates `mtp` → `SAMTPSpeculator` when `SA_MTP=1`;
2. two **model-runner hooks** (`model_runner.py`) — `sa_add_request` / `sa_remove_request`, duck-typed (no-ops for any other speculator).

It deliberately keeps the method string `"mtp"`, so vLLM's MTP machinery and config validation stay untouched — **but that means the anchors are tied to a b12x build.** The patcher **fails loudly** if an anchor isn't found, so **pin your `BASE` image**. A small diagnostic logs SA match-depth every 96 steps (one GPU sync). See `TECHNICAL.md` for the exact integration points if you're porting to another b12x build.

## Credits / License

Ported by [MadeBy561](https://github.com/MadeBy561). Apache-2.0 (see `LICENSE`). Suffix-automaton engine: Baseten [`sa_spec`](https://github.com/basetenlabs/sa_spec) (Apache-2.0). Built on [b12x](https://github.com/lukealonso/b12x) / [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0). See `NOTICE`.
