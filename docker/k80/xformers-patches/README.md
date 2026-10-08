# XFormers patches for sm_37 (Tesla K80, Kepler)

Git-format patches applied to an XFormers source tree at build time, enabling Tesla K80 (sm_37 / Kepler) targets that upstream XFormers does not natively support.

The patches are small and additive — none of them remove or modify upstream functionality. They register sm_37 in the kernel generator and, eventually (Story #32), lower the Python-level minimum compute capability so the dispatcher actually selects the cutlass FMHA op on K80.

## Critical version note — XFormers must be pinned to v0.0.23

Two simultaneous constraints force this pin:

**Constraint 1 — cutlass mem-eff kernels must be in-tree.** After XFormers v0.0.29 (commit `3d947a6`, 2024-12-16, "Remove the cutlass backend of mem-eff, as it is now embedded in PyTorch"), the C++/CUDA mem-eff attention kernels were removed from XFormers entirely — they now live in PyTorch upstream. The Python wrapper `xformers/ops/fmha/cutlass.py` exists in both old and new XFormers, but only the old version has the C++/CUDA kernels behind it. So we need a tag **before v0.0.29**.

**Constraint 2 — must work with the fork's PyTorch.** When this pin was set the fork ran PyTorch 2.0.1; since #84 it runs PyTorch 2.4.1 (`docker/k80/constraints.txt`), still within CUDA 11.4 because the K80 driver R470 caps it there (`docs/port/cuda-11.4-version-pins.md` §5.2). XFormers' PyTorch requirements bumped over time:

| XFormers tag | Date | Required torch |
|---|---|---|
| v0.0.21–v0.0.23 | Aug–Dec 2023 | `>= 1.12` ✓ |
| v0.0.24–v0.0.26 | Jan–Apr 2024 | `>= 2.1` |
| v0.0.27–v0.0.27.post2 | Jul 2024 | `>= 2.2` |
| v0.0.28–v0.0.28.post3 | Sep–Oct 2024 | `>= 2.4` |
| v0.0.29+ | Dec 2024+ | `>= 2.4` (and no kernels — fails Constraint 1) |

**The latest tag satisfying BOTH constraints: `v0.0.23`** (2023-12-06).

With PyTorch 2.0.1, v0.0.23 was the ceiling. On PyTorch 2.4.1 (#84) v0.0.23 still builds and passes `test_cutlass_fp32.py`, and Constraint 2 now allows up to v0.0.28.post3 (the last tag before the v0.0.29 cutlass removal). Moving off v0.0.23 means porting the patches; nothing needs it yet.

## Why patches and not a fork

Same reasoning as `cutlass-patches/`: minimal additive changes are smaller than maintaining a fork, and the patches stay visible in this repository. If patch volume grows, Story #34 (run XFormers' own test suite on K80) will be the natural point to revisit.

## Patches

### `sm37-cc-gate.patch` (Story #32)

Lowers XFormers' Python-level minimum compute capability gate at `xformers/ops/fmha/common.py:253`:

```python
# Before:
CUDA_MINIMUM_COMPUTE_CAPABILITY: Tuple[int, int] = (5, 0)

# After:
CUDA_MINIMUM_COMPUTE_CAPABILITY: Tuple[int, int] = (3, 7)  # Tesla K80 (sm_37) base default
```

**Why:** Per Story 0.2 §3.1's analysis of the dispatcher, the runtime check at `common.py:305` uses this value to reject devices below the threshold. With the default `(5, 0)`, an sm_37 device is refused at dispatch time before any kernel is selected — even if `sm37-generate-kernels.patch` produced kernels for it. Both patches are required for a working dispatch chain.

**What this unlocks:** When Story #33 builds and Story #34 runs, `AttentionOpBase.supports()` will accept an sm_37 device instead of refusing it. Per the same Story 0.2 analysis, individual ops can override `CUDA_MINIMUM_COMPUTE_CAPABILITY` upward (e.g. `flash.py` sets `(8, 0)`); those overrides aren't touched by this patch — the gate only loosens for ops that inherit the base.

**What this does NOT do:**

- Does not by itself produce a working sm_37 XFormers binary. That's Story #33.
- Does not lower the threshold for ops that override it (Flash, Triton, decoder-specific ops). Those ops require sm_70+/sm_80+ for hardware reasons (tensor cores, etc.) that are real, not labelling artifacts. We don't want them to dispatch on K80 — they would crash at kernel launch. The base `cutlassF` op is what we're after, and it inherits the base default we just lowered.

**Applies to:** XFormers v0.0.23.

### `sm37-from-seqlens-device.patch` (Story #35 follow-up)

Backports a `device=` keyword argument to `BlockDiagonalMask.from_seqlens` (and inherited `BlockDiagonalCausalMask.from_seqlens`) at `xformers/ops/fmha/attn_bias.py:550`:

```python
# Before (v0.0.23):
def from_seqlens(cls, q_seqlen, kv_seqlen=None) -> "BlockDiagonalMask":
    ...

# After:
def from_seqlens(cls, q_seqlen, kv_seqlen=None, device=None) -> "BlockDiagonalMask":
    ...
    if device is not None:
        q_seqinfo.to(device)
        if k_seqinfo is not q_seqinfo:
            k_seqinfo.to(device)
    return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)
```

**Why:** vLLM's `vllm/attention/backends/xformers.py` (lines 686, 698, 707, 716) passes `device=query.device` as a keyword argument to `from_seqlens`. The kwarg was added in a later XFormers release; v0.0.23 doesn't accept it, so the call raises `TypeError: BlockDiagonalMask.from_seqlens() got an unexpected keyword argument 'device'` during the engine's profile run, before any kernel launches. Caught by k80-runtime workflow run 24971458508.

**What this unlocks:** Allows vLLM's xformers backend dispatcher to construct attention masks on the GPU without modifying upstream vLLM code. Without it, the K80 fork would have to fork `vllm/attention/backends/xformers.py` to drop the kwarg in four call sites.

**What this does NOT do:**
- Does not change `_SeqLenInfo.to(device)`'s behaviour (already exists in v0.0.23). Just plumbs the device through `BlockDiagonalMask.from_seqlens`.
- Does not affect `BlockDiagonalCausalLocalAttentionMask.from_seqlens` (line 816) — vLLM doesn't call it. If a future vLLM upgrade does, this patch would need a sibling extending the same kwarg there.

**Applies to:** XFormers v0.0.23.

### `sm37-generate-kernels.patch` (Story #31)

Adds `37` to the SM list at `xformers/csrc/attention/cuda/fmha/generate_kernels.py:23`:

```python
# Before:
SM = [50, 70, 75, 80, 100]  # Sm80 kernels support up to Sm100

# After:
SM = [37, 50, 70, 75, 80, 100]  # Sm80 kernels support up to Sm100; Sm37 added for Tesla K80
```

**Why:** Per Story 0.2 §3.3, XFormers' kernel autogen uses this list to materialize per-SM kernel instantiations. Without `37` in the list, `generate_kernels.py` produces zero matching FMHA kernel instantiations for sm_37 — the build silently emits no kernel, and runtime falls through with "no kernel image is available."

**What this unlocks:** When Story #33 builds XFormers from source against our patched CUTLASS (which has `cutlass::arch::Sm37` from `cutlass-patches/sm37-trait.patch`, Phase 1 verified), the autogen will emit a sm_37 instantiation per the SM-list dispatch. The expectation, per Story 0.1 / 0.4, is that those kernels dispatch through CUTLASS's SIMT path and run correctly on K80 — verified empirically by Phase 1's reproducer for standalone GEMM. **Story #33 is the equivalent verification gate for the XFormers-side integration**; this PR alone doesn't prove the XFormers attention kernels work end-to-end on K80.

**What this does NOT do:**

- Does not by itself produce a working sm_37 XFormers binary — that is Story #33.
- Does not patch the Python compute-capability gate at `common.py:253` — that's Story #32, separate one-line patch.
- Does not change XFormers' bundled CUTLASS submodule. XFormers v0.0.23 bundles CUTLASS at commit `e0aaa3c3` (September 2023, pre-v3.5). **Verified: our `cutlass-patches/sm37-trait.patch` applies cleanly to that bundled CUTLASS as-is** — the `arch.h` structure at the patch site is stable across CUTLASS pre-v3.5 → v4.0.0 → HEAD (Sm50 onwards has been unchanged). Story #33's CUTLASS-side work is therefore: apply the existing patch to the submodule, no separate version is needed.

**Applies to:** XFormers v0.0.23.

## Applying the patches

The patches are not yet wired into a build pipeline (Story #33's territory). Manually apply with:

```bash
cd /path/to/xformers-source-at-v0.0.23
git apply /path/to/vllm/docker/k80/xformers-patches/sm37-generate-kernels.patch
```

A dry-run check:

```bash
git apply --check docker/k80/xformers-patches/sm37-generate-kernels.patch
```

### Idempotency

Same as `cutlass-patches/`: patches are not idempotent against `git apply`. Recommended apply patterns for build pipelines:

```bash
# Option A — explicit pre-check via grep marker
grep -q "37, 50, 70, 75, 80, 100" \
    xformers/csrc/attention/cuda/fmha/generate_kernels.py \
  || git apply /path/to/sm37-generate-kernels.patch

# Option B — 3-way merge handles already-applied + minor context drift
git apply -3 /path/to/sm37-generate-kernels.patch

# Option C — GNU patch's --forward flag
patch -p1 --forward < /path/to/sm37-generate-kernels.patch
```

Option B is most durable.

### Automated verification

```bash
make -C docker/k80 verify-xformers-patches
```

Clones XFormers shallow at the pinned tag, runs `git apply --check` per patch. Zero hardware required; matches the same idiom as `verify-cutlass-patches`.

## Pin maintenance

Three scenarios that require maintenance work here:

### Scenario A — XFormers refactor changes file paths

The cutlass mem-eff backend was already removed in v0.0.29; we've already pinned past that risk. But XFormers may further refactor `xformers/csrc/attention/cuda/fmha/` in a future release. Detection: `make verify-xformers-patches` will fail because the target file no longer exists at the new pin.

Recovery options:
1. Stay at v0.0.23 indefinitely (current strategy).
2. Find the new location of the kernel autogen if the refactor moved rather than removed.
3. Bump to the latest pre-removal pin if there's a newer one before v0.0.29.

### Scenario B — upstream adds Sm37 natively

Detection (same idiom as cutlass-patches/README.md scenario B):

```bash
WORK=$(mktemp -d); trap "rm -rf $WORK" EXIT
git clone --depth 1 --branch v0.0.23 https://github.com/facebookresearch/xformers.git $WORK/xformers
git -C $WORK/xformers apply --check        docker/k80/xformers-patches/sm37-generate-kernels.patch || \
    git -C $WORK/xformers apply --check --reverse docker/k80/xformers-patches/sm37-generate-kernels.patch \
    && echo "Sm37 already in XFormers SM list — drop the patch"
```

Recovery: delete the patch and update this README.

### Scenario C — PyTorch pin bump unlocks newer XFormers

If our K80 fork ever upgrades PyTorch past 2.4 (and somehow keeps K80 support — unlikely without a custom build), we could move to a newer XFormers and consume the embedded-in-PyTorch cutlass kernels. Story 0.3 §5.2 documented why this is currently blocked by the R470 driver constraint. If that ever changes:

```bash
# Pivot point: stop maintaining xformers-patches/ entirely.
# The kernels live in PyTorch — patch them there instead.
```

## Conventions for new patches

- Name patches `<feature>-<scope>.patch`.
- Each patch must apply cleanly to v0.0.23 (the pinned version). Test via `git apply --check`.
- Add a section to this README describing what the patch does, why, and what it does *not* do.
- Patches should be additive whenever possible.

## See also

- [`docs/port/flash-attn-mma.md`](../../../docs/port/flash-attn-mma.md) — Story 0.2's analysis (XFormers reference at §3.3).
- [`docs/port/cuda-11.4-version-pins.md`](../../../docs/port/cuda-11.4-version-pins.md) — Story 0.3's pin reasoning, including XFormers HEAD details (now superseded by the v0.0.23 finding above).
- [`docker/k80/cutlass-patches/`](../cutlass-patches/) — sister directory for CUTLASS patches; Phase 1 verified.
