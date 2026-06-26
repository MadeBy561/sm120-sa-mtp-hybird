#!/usr/bin/env python3
"""Idempotently patch the installed b12x vLLM for the SA+MTP hybrid speculator.

- Installs the sa_mtp speculator package into vllm/v1/worker/gpu/spec_decode/sa_mtp/.
- Env-gates the V2 speculator dispatch: SA_MTP=1 -> SAMTPSpeculator, else MTPSpeculator.
  (method stays "mtp" so the rest of the MTP machinery / V2 gate is untouched.)
- Adds duck-typed add_request / remove_request hooks in the V2 model runner
  (no-ops for non-SA speculators).

Run inside the image at build time. Re-runnable (idempotent).
"""
import argparse
import importlib.util
import os
import py_compile
import shutil
import sys

# Locate the installed vllm package dir WITHOUT importing it (no GPU at build time).
_spec = importlib.util.find_spec("vllm")
if _spec is None or not _spec.submodule_search_locations:
    print("[FAIL] cannot locate installed vllm package", file=sys.stderr)
    sys.exit(2)
VLLM = list(_spec.submodule_search_locations)[0]

PATCHES = [
    # ---- V2 dispatch: env-gate MTP -> SA+MTP ----
    (
        "v1/worker/gpu/spec_decode/__init__.py",
        '''    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)''',
        '''    elif speculative_config.method == "mtp":
        import os as _os

        if _os.environ.get("SA_MTP") == "1":
            from vllm.v1.worker.gpu.spec_decode.sa_mtp.speculator import (
                SAMTPSpeculator,
            )

            return SAMTPSpeculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)''',
    ),
    # ---- model runner: add_request hook ----
    (
        "v1/worker/gpu/model_runner.py",
        '''            req_index = self.req_states.req_id_to_index[req_id]

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)''',
        '''            req_index = self.req_states.req_id_to_index[req_id]

            _spec = self.speculator
            if _spec is not None and hasattr(_spec, "sa_add_request"):
                _spec.sa_add_request(
                    req_id, req_index, new_req_data.prompt_token_ids
                )

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)''',
    ),
    # ---- model runner: remove_request hook ----
    (
        "v1/worker/gpu/model_runner.py",
        '''        self.lora_state.remove_request(req_id)
        return True''',
        '''        self.lora_state.remove_request(req_id)
        _spec = self.speculator
        if _spec is not None and hasattr(_spec, "sa_remove_request"):
            _spec.sa_remove_request(req_id, req_idx)
        return True''',
    ),
]


def install_pkg(src: str) -> str:
    dst = os.path.join(VLLM, "v1/worker/gpu/spec_decode/sa_mtp")
    os.makedirs(dst, exist_ok=True)
    for fn in ("__init__.py", "speculator.py"):
        shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
    py_compile.compile(os.path.join(dst, "speculator.py"), doraise=True)
    print(f"[ok] installed sa_mtp package -> {dst}")
    return dst


def apply() -> None:
    for rel, old, new in PATCHES:
        path = os.path.join(VLLM, rel)
        with open(path) as f:
            content = f.read()
        if new in content:
            print(f"[skip] already patched: {rel}")
            continue
        if old not in content:
            print(f"[FAIL] anchor not found in {rel}", file=sys.stderr)
            sys.exit(2)
        with open(path, "w") as f:
            f.write(content.replace(old, new, 1))
        py_compile.compile(path, doraise=True)
        print(f"[ok] patched: {rel}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-pkg", default=None, help="dir with __init__.py + speculator.py")
    args = ap.parse_args()
    print(f"[info] vllm at {VLLM}")
    if args.install_pkg:
        install_pkg(args.install_pkg)
    apply()
    print("[done] SA+MTP patches applied.")
