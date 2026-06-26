# SA + MTP Hybrid speculator — layered onto a b12x vLLM image.
#
# Build:
#   docker build --build-arg BASE=<your-b12x-vllm-image> \
#                --build-arg CUDA_ARCH=120 -t vllm-sa-mtp .
#
# CUDA_ARCH = your GPU compute capability without the dot (120 = Blackwell / RTX PRO 6000,
# 90 = Hopper, 100 = B200). sa_spec pins no CUDA arch, so this is REQUIRED.
ARG BASE
FROM ${BASE}

ARG CUDA_ARCH=120
ARG SA_SPEC_REPO=https://github.com/basetenlabs/sa_spec
ARG SA_SPEC_REF=main
# Extra cmake defs, e.g. -DC_MAX_SLOTS=64 (batch>32) or -DC_MAX_SEQUENCE_LENGTH=... (ctx>256k).
ARG SA_CMAKE_EXTRA=

# 1) Baseten's suffix-automaton engine (Apache-2.0), compiled for your GPU arch.
RUN pip install --no-cache-dir nanobind scikit-build-core \
 && git clone "${SA_SPEC_REPO}" /tmp/sa_spec \
 && git -C /tmp/sa_spec checkout "${SA_SPEC_REF}" \
 && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} ${SA_CMAKE_EXTRA}" \
      pip install --no-deps /tmp/sa_spec \
 && python3 -c "import sa_spec; print('sa_spec OK, SA_SPEC_THRESH =', sa_spec.SA_SPEC_THRESH)"

# 2) The hybrid speculator package + the env-gated patch (idempotent, fails loud on anchor mismatch).
COPY sa_mtp/ /tmp/sa_mtp/
COPY apply_sa_patches.py /tmp/apply_sa_patches.py
RUN python3 /tmp/apply_sa_patches.py --install-pkg /tmp/sa_mtp
