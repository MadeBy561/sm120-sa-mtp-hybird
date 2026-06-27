# SPDX-License-Identifier: Apache-2.0
# SA+MTP hybrid speculator for the b12x V2 model runner (Baseten threshold-switch design).
#
# Per request: stock MTP drafts a SHORT prefix (MTP_DRAFT_DEPTH forwards, default 5 -- MTP's
# accept dies past ~5 positions anyway), and the suffix automaton (sa_spec) overlays a LONG
# copied draft (full num_spec width) on any row whose context match >= SA_SPEC_THRESH.
# The rejection sampler verifies whatever draft ids land in the batch, so this is lossless
# regardless of token source (greedy: accept iff draft == target argmax).
#
# WHY THE CAP + LEAN-ON-MISS (the important part): num_spec must be wide (16) so SA can copy long
# runs. Two costs are decoupled: (1) MTP_DRAFT_DEPTH caps the MTP DRAFT forwards (positions >~5 are
# 100% rejected anyway); (2) LEAN-ON-MISS returns only the MTP-cap width on an SA-miss so the VERIFY
# forward is ~plain-MTP cost too -- full S width is returned ONLY when SA actually fires. Net: SA is
# a pure bonus with ZERO novel-decode regression. num_spec = SA width; MTP_DRAFT_DEPTH = miss width.
#
# Notes: SA overlay runs eagerly (outside cudagraph capture); any failure -> pure MTP (lossless).
# accepted-token reconstruction is sync-free. No per-step CPU sync.

import os

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

logger = init_logger(__name__)

try:
    import sa_spec

    _SA_IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    sa_spec = None
    _SA_IMPORT_ERR = e

_SA_MAX_SEQ = int(os.environ.get("SA_SPEC_MAX_SEQ", str(262144))) - 16


class SAMTPSpeculator(MTPSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if sa_spec is None:
            raise ImportError(
                f"sa_spec is required for the 'sa_mtp' speculator: {_SA_IMPORT_ERR}"
            )

        S = self.num_speculative_steps
        R = self.max_num_reqs
        self.sa_thresh = sa_spec.SA_SPEC_THRESH
        self.sa_disabled = not (self.sa_thresh < float("inf")) or S < 2
        # Isolation knob: SA_SPEC_THRESH=9999999 -> SA machinery loads but the per-step overlay is
        # SKIPPED (propose returns the raw MTP draft). Measures the pure 16-verify cost WITHOUT the
        # overlay, so we can tell if the novel tax is the verify (inherent) or the overlay (fixable).
        if self.sa_thresh > 1_000_000:
            self.sa_disabled = True
        # MTP runs at most this many draft forwards/step (it's dead past ~5). SA keeps full S.
        self._mtp_cap = max(1, min(int(os.environ.get("MTP_DRAFT_DEPTH", "5")), S))
        # LEAN-ON-MISS: width of the draft RETURNED last step (S on a match, _mtp_cap on a miss).
        # The accepted-length reconstruction must use THIS, not the buffer width S.
        self._last_width = S

        self.sa_depth = torch.zeros(R, dtype=torch.int32, device=device)
        self.sa_draft = torch.zeros(R, S, dtype=torch.int32, device=device)
        self.sa_accepted = torch.zeros(R, S + 1, dtype=torch.int32, device=device)
        self.sa_accepted_lens = torch.zeros(R, dtype=torch.int32, device=device)
        self.sa_prev_draft = torch.zeros(R, S, dtype=torch.int64, device=device)
        self.sa_first = torch.zeros(R, dtype=torch.int32, device=device)
        self._sa_arange = torch.arange(S + 1, dtype=torch.int64, device=device)

        self._uid_of_reqid: dict[str, int] = {}
        self._last_uids: list[int] | None = None  # cache: skip redundant sa_spec.prepare()
        self._next_uid = 1
        self._sa_err_logged = False
        self._sa_debug = os.environ.get("SA_DEBUG") == "1"
        self._sa_calls = 0  # for periodic depth diagnostics
        self._sa_depth_acc = torch.zeros((), dtype=torch.float32, device=device)
        self._sa_overlay_acc = torch.zeros((), dtype=torch.float32, device=device)
        logger.info(
            "SAMTPSpeculator active: S=%d, MTP_DRAFT_DEPTH=%d, SA_SPEC_THRESH=%s, disabled=%s",
            S, self._mtp_cap, self.sa_thresh, self.sa_disabled,
        )

    # ---- model-runner hooks -------------------------------------------------
    def sa_add_request(self, req_id: str, req_index: int, prompt_token_ids) -> None:
        if self.sa_disabled:
            return
        self._uid_of_reqid.pop(req_id, None)
        if not prompt_token_ids:
            return
        if len(prompt_token_ids) > _SA_MAX_SEQ:
            self.sa_first[req_index] = 0
            return
        uid = self._next_uid
        self._next_uid += 1
        sa_spec.add_request(uid, list(prompt_token_ids))
        self._uid_of_reqid[req_id] = uid
        self.sa_first[req_index] = 1

    def sa_remove_request(self, req_id: str, req_index: int) -> None:
        if self.sa_disabled:
            return
        self._uid_of_reqid.pop(req_id, None)
        if req_index is not None and 0 <= req_index < self.max_num_reqs:
            self.sa_first[req_index] = 0

    def _log_sa_err(self, e: Exception) -> None:
        if not self._sa_err_logged:
            self._sa_err_logged = True
            logger.warning("SA error (falling back to pure MTP this run): %r", e)

    # ---- cap MTP draft depth SAFELY (it dies past ~5; SA covers the long tail on matches) ----
    # The old cap-swap mutated self.num_speculative_steps mid-draft. But the parent passes that value
    # to update_draft_inputs as the draft_tokens buffer STRIDE -- so stride-cap writes into a stride-S
    # buffer corrupted the draft -> non-deterministic lossy output. Fix: replicate the parent loop
    # body verbatim and ONLY shrink the range; never touch num_speculative_steps. Cuts the wasted
    # forwards cap..S (~25 t/s -> ~41 t/s novel) with the stride intact.
    def _multi_step_decode(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        cap = min(self._mtp_cap, self.num_speculative_steps)
        if cap >= self.num_speculative_steps:
            return super()._multi_step_decode(
                num_reqs, skip_attn, batch_desc, num_tokens_across_dp
            )
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, cap):
            self.active_num_reqs.fill_(num_reqs)
            if not skip_attn and (self.advance_draft_positions or step == 1):
                with record_function_or_nullcontext(
                    "vllm:v2/speculator/decode_step/prepare_attn"
                ):
                    slot_mappings = self.block_tables.compute_slot_mappings(
                        idx_mapping, query_start_loc, positions, batch_desc.num_tokens
                    )
                    slot_mappings_by_layer = build_slot_mappings_by_layer(
                        slot_mappings, self.kv_cache_config
                    )
                    attn_metadata = self._build_draft_attn_metadata(
                        num_reqs=num_reqs,
                        num_reqs_padded=batch_desc.num_reqs or num_reqs,
                        num_tokens_padded=batch_desc.num_tokens,
                    )
            self.current_draft_step.fill_(step)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                with record_function_or_nullcontext(
                    "vllm:v2/speculator/decode_step/generate"
                ):
                    self._generate_draft(
                        num_reqs,
                        batch_desc.num_tokens,
                        attn_metadata,
                        slot_mappings_by_layer,
                        num_tokens_across_dp=num_tokens_across_dp,
                        cudagraph_runtime_mode=batch_desc.cg_mode,
                    )

    # ---- propose: MTP (capped) then overlay the SA copy where it matches ----
    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        draft = super().propose(
            input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
            num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
            num_tokens_across_dp=num_tokens_across_dp, dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run, mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
        # CRITICAL: _last_width must still hold LAST step's returned width when _sa_overlay reads it
        # (the accepted-length reconstruction L = clamp(W - num_rejected) depends on it). Resetting it
        # here was THE bug: every miss-step reconstructed against S instead of the real lean width ->
        # wrong accepted tokens -> the SA automaton learned garbage -> output degenerated into a self-
        # similar loop -> SA then "matched" that loop and fired on NOVEL text (overlay ~0.95 instead of
        # ~0) -> wide verify every step -> the base-decode tax. _sa_overlay reads the prior value then
        # commits the new one; the early / fallback paths below return full S-wide -> set _last_width=S.
        if self.sa_disabled or dummy_run or is_profile or input_batch.num_reqs == 0:
            self._last_width = self.num_speculative_steps
            return draft
        try:
            return self._sa_overlay(
                input_batch, num_sampled, num_rejected, last_sampled, draft
            )
        except Exception as e:  # noqa: BLE001
            self._log_sa_err(e)
            self._last_width = self.num_speculative_steps
            return draft

    def _sa_overlay(self, input_batch, num_sampled, num_rejected, last_sampled, draft):
        B = input_batch.num_reqs
        uids = []
        for rid in input_batch.req_ids[:B]:
            u = self._uid_of_reqid.get(rid)
            if u is None:
                return draft
            uids.append(u)

        device = self.draft_tokens.device
        S = self.num_speculative_steps
        idx = input_batch.idx_mapping[:B].long()

        # accepted = prev_draft[:L] + [bonus]; L = clamp(S - num_rejected, 0, S); first -> 0.
        prev = self.sa_prev_draft.index_select(0, idx)
        first = self.sa_first.index_select(0, idx).bool()
        nrej = num_rejected[:B].reshape(-1).to(torch.int64)
        gen = num_sampled[:B].reshape(-1) > 0
        bonus = last_sampled.index_select(0, idx).reshape(B, -1)[:, 0].to(torch.int32)

        # LEAN-ON-MISS: accepted count is relative to the width ACTUALLY returned last step.
        W = self._last_width
        L = torch.clamp(W - nrej, min=0, max=W)
        L = torch.where(first, torch.zeros_like(L), L)
        prefix_mask = self._sa_arange[:S].unsqueeze(0) < L.unsqueeze(1)

        acc = self.sa_accepted[:B]
        acc.zero_()
        acc[:, :S] = torch.where(
            prefix_mask, prev.to(torch.int32), torch.zeros_like(prev, dtype=torch.int32)
        )
        acc.scatter_(1, L.unsqueeze(1), bonus.unsqueeze(1))
        zeros_b = torch.zeros(B, dtype=torch.int32, device=device)
        self.sa_accepted_lens[:B].copy_(torch.where(gen, (L + 1).to(torch.int32), zeros_b))
        self.sa_first.index_copy_(
            0, idx, torch.where(gen, zeros_b, self.sa_first.index_select(0, idx))
        )

        # prepare() sets the active automatons for this batch; for a steady single/again-same batch
        # the uids don't change, so calling it every step is pure redundant per-step overhead.
        if uids != self._last_uids:
            sa_spec.prepare(uids)
            self._last_uids = list(uids)
        sa_spec.extend(self.sa_depth[:B], self.sa_draft[:B], acc, self.sa_accepted_lens[:B])

        # DIAG: accumulate match depth + overlay rate on GPU; sync+log every 96 calls.
        # overlay_rate ~1 => SA matches every step (bug is in the splice); ~0.2 => SA keeps
        # losing the match between spikes (feedback drift).
        self._sa_calls += 1
        if self._sa_debug and self._sa_calls <= 80:
            # Per-step reconstruction trace. On forced repetition, depth should CLIMB toward the
            # line length each rep; if it resets to ~2-3 every few steps, the accepted-length L /
            # acclen are drifting the automaton position. W=last returned width, nrej=rejected.
            logger.info(
                "SA-DBG c=%d W=%d nrej=%d L=%d acclen=%d depth=%d first=%d draft=%s",
                self._sa_calls, int(W), int(nrej[0].item()), int(L[0].item()),
                int(self.sa_accepted_lens[0].item()), int(self.sa_depth[0].item()),
                int(first[0].item()), self.sa_draft[0, :6].tolist(),
            )
        self._sa_depth_acc += self.sa_depth[:B].float().mean()
        self._sa_overlay_acc += (self.sa_depth[:B] >= self.sa_thresh).float().mean()
        if self._sa_calls % 96 == 0:
            logger.info(
                "SA-DIAG/96 mean_depth=%.1f overlay_rate=%.2f",
                (self._sa_depth_acc / 96).item(), (self._sa_overlay_acc / 96).item(),
            )
            self._sa_depth_acc.zero_()
            self._sa_overlay_acc.zero_()

        # overlay the SA copy on rows that matched >= threshold
        mask = (self.sa_depth[:B] >= self.sa_thresh).unsqueeze(1)
        out = self.draft_tokens[:B]
        out.copy_(torch.where(mask, self.sa_draft[:B].to(torch.int64), out))
        self.sa_prev_draft.index_copy_(0, idx, out)
        # FIXED-16 (no lean). The lean's NARROW verify runs EAGER (its token-count isn't a captured
        # cudagraph size) -> the SA-sits-out floor craters to ~23 t/s. The CAPTURED 16-wide verify is
        # the fastest verify available -> a stable ~43 floor. Narrowing the verify backfires, so we
        # always return full S; W == the real verify width keeps the reconstruction consistent.
        self._last_width = S
        return out
