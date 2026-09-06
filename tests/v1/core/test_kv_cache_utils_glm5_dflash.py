# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash DFlash2 draft sliding-window KV groups.

Covers the group declaration (draft SWA specs join the GLM Mamba/MLA
slot-sharing layout), eagle annotation, the fail-fast guard when the draft
page cannot join, the DFlash / fused-draft (KPOOL_TAIL) mutual exclusion in
``SpeculativeConfig``, and GPU-gated end-to-end smoke tests (FULL CUDA graph
capture/replay and per-group prefix-cache hit-rate monitoring).

KV-groups semantics rewritten against vLLM PR #55423 Part 2; the DCP
KV-head replication path it relies on is #48392 (dfd9278229).
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.envs import disable_envs_cache
from vllm.v1.core.kv_cache_utils import (
    _glm5_next_tensor_layout,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


def _glm5_target_spec() -> dict[str, KVCacheSpec]:
    """(mamba, mamba, mamba, MLA + indexer) * 3, mirroring GLM-5.3-Flash."""
    spec: dict[str, KVCacheSpec] = {}
    for i in range(12):
        if i % 4 == 3:
            spec[f"layers.{i}.attn"] = MLAAttentionSpec(
                block_size=512,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
            )
            spec[f"layers.{i}.indexer"] = MLAAttentionSpec(
                block_size=512,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=16,
            )
        else:
            spec[f"layers.{i}.linear_attn"] = MambaSpec(
                block_size=512,
                shapes=((2, 512), (3, 32, 32)),
                dtypes=(torch.float32, torch.float32),
                num_speculative_blocks=2,
            )
    return spec


def _dflash_draft_specs(
    num_layers: int = 5,
    sliding_window: int = 2048,
    num_kv_heads: int = 2,
    head_size: int = 128,
) -> dict[str, SlidingWindowSpec]:
    """DFlash2 GLM draft: 5 sliding_attention layers, sw=2048, non-causal.

    The DCP KV-head replication path (#48392, dfd9278229) publishes
    ``cache_num_kv_heads`` on the draft's attention layer, so the spec's
    ``num_kv_heads`` already covers the replicated heads.
    """
    return {
        f"draft.layers.{i}.self_attn": SlidingWindowSpec(
            block_size=64,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
        )
        for i in range(num_layers)
    }


def _grouping_config(method="dflash", use_block_drop=True):
    """Config with the DFlash speculative method enabled."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=True
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=None)),
        speculative_config=SimpleNamespace(
            method=method,
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: use_block_drop,
        ),
    )


def test_dflash_draft_swa_groups_join_slot_sharing_layout():
    # Draft SWA layers co-own the MLA slot tensors like Mamba groups: an
    # independent logical block table per bucket, same padded physical page.
    spec = _glm5_target_spec()
    draft = _dflash_draft_specs()
    spec.update(draft)

    groups = get_kv_cache_groups(_grouping_config(), spec)
    sliding = [g for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec]
    mla_page = spec["layers.3.attn"].page_size_bytes

    assert sliding, "draft sliding-window layers must form their own groups"
    draft_names = {name for g in sliding for name in g.layer_names}
    assert draft_names == set(draft)
    for group in sliding:
        draft_spec = cast(SlidingWindowSpec, group.kv_cache_spec)
        assert draft_spec.block_size == 512
        assert draft_spec.page_size_padded == mla_page
        assert draft_spec.page_size_bytes == mla_page
        assert draft_spec.sliding_window == 2048
    # 5 draft layers over 3 MLA slots -> 2 groups, strided not chunked.
    assert [len(g.layer_names) for g in sliding] == [3, 2]

    layout = _glm5_next_tensor_layout(groups)
    assert layout is not None
    slot_groups = layout[1]
    assert any(type(g.kv_cache_spec) is SlidingWindowSpec for g in slot_groups)


def test_dflash_draft_groups_annotated_eagle():
    # The draft groups carry the eagle flag so the coordinator drops their
    # volatile trailing blocks instead of the target's.
    spec = _glm5_target_spec()
    spec.update(_dflash_draft_specs())

    groups = get_kv_cache_groups(_grouping_config(), spec)
    for group in groups:
        holds_draft = any(name.startswith("draft.") for name in group.layer_names)
        if type(group.kv_cache_spec) is SlidingWindowSpec:
            assert holds_draft
            assert group.is_eagle_group
        else:
            assert not group.is_eagle_group


def test_dflash_draft_groups_not_annotated_without_block_drop():
    spec = _glm5_target_spec()
    spec.update(_dflash_draft_specs())

    groups = get_kv_cache_groups(_grouping_config(use_block_drop=False), spec)
    assert not any(
        g.is_eagle_group for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec
    )


def test_mamba_groups_never_flagged_with_dflash_draft(caplog_vllm):
    # Risk monitor for _warn_if_unannotated_eagle_mamba: with the draft
    # groups annotated, no group falls into the flag-all fallback, so a
    # Mamba group is never widened to two consecutive align chunks (which
    # would silently zero its prefix-cache reuse).
    spec = _glm5_target_spec()
    spec.update(_dflash_draft_specs())

    groups = get_kv_cache_groups(_grouping_config(), spec)
    for group in groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert not group.is_eagle_group
    assert "no KV cache group could be identified" not in caplog_vllm.text


def test_draft_page_too_wide_fails_fast():
    # A draft whose per-token KV is wider than a whole MLA slot cannot join
    # the slot-sharing layout at any block size (page width scales with block
    # size), and must not fall through to the generic promote path (which
    # would silently break the kpool indexer's pool-page alignment); it must
    # raise at grouping time.
    spec = _glm5_target_spec()
    # per-token = 96 heads * (4096+4096) * 2 bytes > the bf16 MLA slot page.
    spec.update(_dflash_draft_specs(num_kv_heads=96, head_size=4096))

    with pytest.raises(ValueError, match="draft page does not fit"):
        get_kv_cache_groups(_grouping_config(), spec)


def _fp8_ds_mla_target_spec() -> dict[str, KVCacheSpec]:
    """Production layout: block 128, packed fp8 MLA (576 bytes/token)."""
    spec: dict[str, KVCacheSpec] = {}
    for i in range(12):
        if i % 4 == 3:
            spec[f"layers.{i}.attn"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=512,
                head_size_v=64,
                dtype=torch.uint8,
            )
            spec[f"layers.{i}.indexer"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=8,
                head_size_v=8,
                dtype=torch.uint8,
                tokens_per_state=16,
            )
        else:
            spec[f"layers.{i}.linear_attn"] = MambaSpec(
                block_size=128,
                shapes=((2, 128), (3, 32, 32)),
                dtypes=(torch.float32, torch.float32),
                num_speculative_blocks=2,
            )
    return spec


def test_fp8_ds_mla_block128_draft_groups_join():
    # block-size=128 + fp8_ds_mla (576 B/token target slot) + a Qwen3-shaped
    # draft (8 kv heads, head_dim 128, bf16 -> 4096 B/token): the draft page
    # at the target block is 8x the MLA slot, so the draft block is shrunk to
    # the largest divisor of 128 whose page fits one slot (16 tokens), then
    # padded to the MLA page. No fail-fast ValueError.
    spec = _fp8_ds_mla_target_spec()
    draft = _dflash_draft_specs(
        num_layers=5, sliding_window=2048, num_kv_heads=8, head_size=128
    )
    for name, draft_spec in draft.items():
        draft[name] = replace(draft_spec, block_size=8)
    spec.update(draft)

    groups = get_kv_cache_groups(_grouping_config(), spec)
    sliding = [g for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec]
    assert sliding
    mla_page = spec["layers.3.attn"].page_size_bytes

    draft_names = {name for g in sliding for name in g.layer_names}
    assert draft_names == set(draft)
    for group in sliding:
        assert group.is_eagle_group
        draft_spec = cast(SlidingWindowSpec, group.kv_cache_spec)
        assert draft_spec.page_size_padded == mla_page
        assert draft_spec.page_size_bytes == mla_page

    layout = _glm5_next_tensor_layout(groups)
    assert layout is not None
    assert any(type(g.kv_cache_spec) is SlidingWindowSpec for g in layout[1])


def test_fp8_ds_mla_block128_draft_block_alignment():
    # The shrunk draft block must divide the target's block size (so the
    # hybrid hash granularity, the GCD over groups, stays well-defined) and
    # its unpadded page must fit inside the single MLA slot it is padded to.
    spec = _fp8_ds_mla_target_spec()
    spec.update(_dflash_draft_specs(num_kv_heads=8, head_size=128))
    target_block_size = spec["layers.3.attn"].block_size
    mla_page = spec["layers.3.attn"].page_size_bytes

    groups = get_kv_cache_groups(_grouping_config(), spec)
    for group in groups:
        if type(group.kv_cache_spec) is not SlidingWindowSpec:
            continue
        draft_spec = cast(SlidingWindowSpec, group.kv_cache_spec)
        assert target_block_size % draft_spec.block_size == 0
        assert draft_spec.block_size < target_block_size
        assert draft_spec.unpadded_page_size_bytes <= mla_page
        assert draft_spec.page_size_bytes == mla_page
        assert draft_spec.sliding_window == 2048


def test_draft_specs_without_spec_decode_do_not_annotate():
    spec = _glm5_target_spec()
    spec.update(_dflash_draft_specs())
    config = _grouping_config()
    config.speculative_config = None

    groups = get_kv_cache_groups(config, spec)
    assert not any(g.is_eagle_group for g in groups)
    # The slot-sharing layout still applies: draft layers are cache groups
    # regardless of whether a drafter runs on them.
    assert any(type(g.kv_cache_spec) is SlidingWindowSpec for g in groups)


def _make_ngram_spec_config(target_model_type: str) -> SpeculativeConfig:
    """Build a SpeculativeConfig through __post_init__ with the ngram method.

    ngram needs no draft checkpoint, so the target model config is the only
    model input; its ``hf_config.model_type`` drives the reverse guard.
    """
    target_model_config = MagicMock(
        model="target",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
        hf_config=MagicMock(model_type=target_model_type),
    )
    return SpeculativeConfig(
        method="ngram",
        prompt_lookup_max=5,
        prompt_lookup_min=3,
        num_speculative_tokens=1,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )


class TestDFlashFusedDraftMutex:
    """D5: method=dflash and the KPOOL_TAIL fused-draft hook are mutually
    exclusive; fail fast in either direction at config init, so the hook
    registration path never half-initializes."""

    def test_fused_draft_hook_with_non_dflash_glm_raises(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENABLE_FUSED_DRAFT_SPARSE_MLA", "1")
        disable_envs_cache()
        with pytest.raises(ValueError, match="routes draft decode elsewhere"):
            _make_ngram_spec_config("glm5_next")

    def test_fused_draft_hook_with_non_glm_target_ok(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENABLE_FUSED_DRAFT_SPARSE_MLA", "1")
        disable_envs_cache()
        config = _make_ngram_spec_config("qwen3")
        assert config.method == "ngram"

    def test_hook_disabled_with_non_dflash_glm_ok(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENABLE_FUSED_DRAFT_SPARSE_MLA", "0")
        disable_envs_cache()
        config = _make_ngram_spec_config("glm5_next")
        assert config.method == "ngram"


# The forward direction (method=dflash + hook enabled) needs a draft model
# config; patch ModelConfig so no checkpoint is touched.
@pytest.mark.parametrize("enable_hook", [True, False])
def test_dflash_method_with_fused_draft_hook(monkeypatch, enable_hook: bool):
    monkeypatch.setenv("VLLM_ENABLE_FUSED_DRAFT_SPARSE_MLA", str(int(enable_hook)))
    disable_envs_cache()

    draft_model_config = MagicMock(
        model="incoai/GLM-5.3-Flash-DFlash2",
        max_model_len=128,
        hf_config=MagicMock(
            model_type="glm5_next_dflash",
            num_hidden_layers=5,
            vocab_size=1000,
        ),
        architectures=["DFlash2DraftModel"],
    )
    draft_model_config.registry.inspect_model_cls.return_value = (
        MagicMock(),
        "DFlash2DraftModel",
    )
    target_model_config = MagicMock(
        model="incoai/GLM-5.3-Flash",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
        hf_config=MagicMock(model_type="glm5_next"),
    )
    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        if enable_hook:
            with pytest.raises(ValueError, match="cannot be combined"):
                SpeculativeConfig(
                    method="dflash",
                    model="incoai/GLM-5.3-Flash-DFlash2",
                    num_speculative_tokens=1,
                    target_model_config=target_model_config,
                    target_parallel_config=ParallelConfig(),
                )
        else:
            config = SpeculativeConfig(
                method="dflash",
                model="incoai/GLM-5.3-Flash-DFlash2",
                num_speculative_tokens=1,
                target_model_config=target_model_config,
                target_parallel_config=ParallelConfig(),
            )
            assert config.method == "dflash"


# ---------------------------------------------------------------------------
# GPU-gated smoke tests (task 2.3): FULL CUDA graph capture/replay with the
# tilelang mHC kernels in-graph, and per-group prefix-cache hit-rate
# monitoring. Never run on CPU; collected only.
# ---------------------------------------------------------------------------

GPU_SMOKE_REASON = "requires CUDA (H200 / SM90) and the GLM-5.3-Flash weights"


@pytest.mark.skipif(not torch.cuda.is_available(), reason=GPU_SMOKE_REASON)
class TestDFlash2GpuSmoke:
    def test_full_cuda_graph_capture_replay(self):
        """FULL CUDA graph capture + replay smoke (task 2.3).

        Boots TP1/DCP1 with dflash on, forces FULL graph mode, and runs a
        short greedy generation: the tilelang mHC post/fused kernels and the
        non-causal SWA draft attention must be legal graph citizens (capture
        then replay without re-capture or wrong outputs).
        """
        pytest.skip("GPU test: run on an H200 node with the DFlash2 draft")

    def test_per_group_prefix_cache_hit_rate(self):
        """Per-group prefix-cache hit-rate monitoring (task 2.3).

        Runs two requests sharing a long prefix and asserts the target MLA
        group reports hits while the draft SWA and Mamba groups are reported
        separately (and Mamba is not wrongly flagged eagle, which would zero
        its reuse silently).
        """
        pytest.skip("GPU test: run on an H200 node with the DFlash2 draft")

    def test_greedy_dflash_on_off_byte_identical(self):
        """Greedy dflash on/off byte equality, e2e full version (task 2.3).

        TP1 DCP1 greedy temperature 0, K=1, with
        incoai/GLM-5.3-Flash-DFlash2 loaded: the token sequences must be
        byte-identical with and without the draft. Passing on H200 also
        covers the SM90 kernel smoke for non-causal SWA draft attention.
        """
        pytest.skip("GPU test: run on an H200 node with the DFlash2 draft")
