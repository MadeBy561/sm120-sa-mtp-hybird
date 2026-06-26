# SA+MTP Hybrid → b12x V2 Model Runner — Integration Map

Goal: port Baseten's suffix-automaton (github.com/basetenlabs/sa_spec, Apache-2.0) into b12x vLLM
(lukealonso/b12x @ aecc88f) as a new V2 speculator `sa_mtp` that wraps MTP with a threshold switch.
SA gives accept-len 10+ on code → path to high single-stream throughput on a 4× RTX PRO 6000 (sm120) box.
References below are to the b12x vLLM source tree (`worker/gpu/spec_decode`, `config/`, `model_runner.py`).

## The unlock (why this works)
- **Rejection sampler is source-agnostic + lossless.** `worker/gpu/spec_decode/rejection_sampler.py:123`
  `draft_sampled = input_batch.input_ids[input_batch.logits_indices]`; `draft_logits` is `None` for
  non-probabilistic → greedy "accept iff draft == target argmax" (`rejection_sampler_utils.py:225-253`).
  SA-copied ids placed in `input_ids` verify identically to neural drafts. **No verify changes.**
- `combine_sampled_and_draft_tokens` (`worker/gpu/input_batch.py:355-396`) derives
  `num_speculative_steps = draft_tokens.shape[-1]` and writes draft ids into input_ids → logits_indices.

## V2 gate (config/vllm.py)
- GLM launched with `VLLM_USE_V2_MODEL_RUNNER=1` → `use_v2_model_runner` returns True at vllm.py:566
  (env short-circuits), so `_validate_v2_model_runner` (2123-2137) HARD-RAISES on unsupported — NO V1 fallback.
- Allow-list: `vllm.py:2057` → `elif method not in ("eagle","eagle3","mtp","dflash"): unsupported.append(...)`.
  **PATCH: add "sa_mtp" to that tuple.**
- V1-fallback (`vllm.py:582-586`) only fires when env UNSET → irrelevant for GLM (and V1 illegal-mems anyway).

## Method routing (config/speculative.py)
- `SpeculativeMethod` Literal @ 155-164 (incl "suffix","custom_class",EagleModelTypes,NgramGPUTypes).
  **PATCH: add "sa_mtp".**
- MTP types (deepseek_mtp/glm4_moe_mtp/…) normalize → "mtp" @ 737-741; "mtp".model = target model @ 744-752.
- `use_eagle()` @ 1288-1289 = method in ("eagle","eagle3","mtp","dflash") → eagle-style KV/hidden drafting.
  **PATCH: add "sa_mtp"** so it inherits the MTP drafting path. Route sa_mtp like mtp in __post_init__.

## V2 speculator interface (worker/gpu/spec_decode/)
- `BaseSpeculator` ABC @ speculator.py:29-69 → 3 methods: `init_cudagraph_manager`, `capture`, `propose(...)`.
- `DraftModelSpeculator` @ speculator.py:72-324: allocates `self.draft_tokens` [max_num_reqs, num_spec_steps] int64
  (118-123); num_speculative_steps = num_speculative_tokens (80). `MTPSpeculator`(mtp/speculator.py:12) →
  `AutoRegressiveSpeculator.propose` (autoregressive/speculator.py:155-346) returns
  `self.draft_tokens[:num_reqs]` shape [num_reqs, num_spec_steps] (346) or [:, :1] if steps==1 (310).
  Uses FULL cudagraph replay `run_fullgraph` (281-289).
- Dispatch: `worker/gpu/spec_decode/__init__.py:8-40` `init_speculator` — chain of elifs on method.
  **PATCH @ :23-ish: `elif method == "sa_mtp": return SAMTPSpeculator(vllm_config, device)`.**

## Model-runner hooks (worker/gpu/model_runner.py = V2 runner)
- `sa_spec.add_request(req_id_int, prompt_token_ids)` → in `add_requests` @ 817-828 (prompt avail @ 819/828).
  Called from step @ 1179. **This is the one runner edit needed.**
- `sa_spec.prepare(active_ids)` + `extend(...)` → inside SAMTPSpeculator.propose (cleanest; runner calls
  speculator.propose @ 1561, stores draft_tokens @ 1581). num_rejected/accepted available @ 1564.

## sa_spec API (sa_spec_src/)
- Python: `add_request(request_id:int, tokens:list[int])`, `prepare(request_ids:list[int])`,
  `extend(depth_out, draft_out, accepted_in, accepted_lens_in)` — all int32 CUDA, c-contiguous.
  Shapes: depth_out [B], draft_out [B, draft_len], accepted_in [B, draft_len+1], accepted_lens_in [B].
  `SA_SPEC_THRESH` env default 8; `DISABLE_SA_SPEC=1` stubs out (→ clean A/B vs pure MTP).
- `extend` CUDA-graph-capturable (test/test_api.py test_cuda_graph), kernel <<<batch,1>>> on torch stream (api.cu:81).
- Compile consts config.hpp:13-25: MAX_SEQUENCE_LENGTH=262144 (256k), MAX_SLOTS=32 (max batch).
  Override: `-DC_MAX_SEQUENCE_LENGTH=`, `-DC_MAX_SLOTS=`.
- Build: scikit-build-core + nanobind, C++23/CUDA20, module `_sa_spec_impl`. NO arch pinned anywhere
  (no CMAKE_CUDA_ARCHITECTURES / -gencode / __CUDA_ARCH__). **sm120: `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120"`.**

## Arctic = V1-only (DEAD END for us, confirmed)
- arctic_inference/vllm/ plugin (plugin.py:24-45) monkeypatches V1 `gpu_model_runner.propose_draft_token_ids`
  (model_runner.py:99) — that method does NOT exist on the V2 runner. So arctic's suffix can't run on our V2 GLM.
  We do NOT reuse arctic; we go V2-native via init_speculator + BaseSpeculator.

## Build plan
1. [#8] Build sa_spec for sm120 in the b12x toolchain; verify import + run its cudagraph pytest. ← DE-RISK FIRST
2. [#9] ADD worker/gpu/spec_decode/sa_mtp/{__init__.py,speculator.py}; PATCH the 4 sites above.
3. [#10] Layer sa_spec .so + patches onto prebuilt b12x image (Dockerfile, no full rebuild); boot GLM-5.2-REAP.
   Bring up cudagraph_mode=PIECEWISE first; set -DC_MAX_SLOTS ≥ MAX_NUM_SEQS.
4. [#11] Warm single-req @ Max on codegen; measure accept-len + t/s vs MTP. Target accept-len 10+/200+ t/s.
   A/B with DISABLE_SA_SPEC=1 (README: thresh=inf ⇒ on par with vanilla MTP).

## Biggest risk
cudagraph capture of sa_spec.extend under GLM FULL-replay (illegal-mem history per project memory).
De-risk: validate sa_spec's own cudagraph test on sm120 (#8) BEFORE writing the speculator; bring up PIECEWISE;
MAX_SLOTS ≥ MAX_NUM_SEQS; gate via DISABLE_SA_SPEC for clean bisection.
