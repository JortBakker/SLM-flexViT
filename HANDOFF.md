# FlexGPT / FlexLLaMA Handoff Document

## Project Overview

FlexGPT and FlexLLaMA are flexible-width decoder-only transformer language models. Width is controlled by nested weight slicing across 3 levels:
- Level 0 (smallest): hidden=384, heads=6, mlp=1536
- Level 1 (medium):   hidden=512, heads=8, mlp=2048
- Level 2 (largest):  hidden=768, heads=12, mlp=3072
- 12 layers fixed across all levels

All levels share the same weight tensors (level 0 uses W[:384,:384], level 2 uses full W[:768,:768]).

FlexLLaMA uses the same sizing but with RoPE attention (`flex_modules/rope_attention.py`) and LLaMA-style architecture, trained on FineWeb-Edu with vocab_size=32000.

---

## Current Experiments

### 1. `flexgpt,wikitext103.kd_from_gpt2` — KD from frozen GPT-2 teacher
- **Status**: Currently running on Snellius (or just crashed — see bug below)
- **Trainer**: `FlexLMKDTrainer` with frozen GPT-2 Small teacher
- **Loss**: pure KL divergence (`kd_lambda=1.0`), no CE — teacher signal only
- **Purpose**: Preserve GPT-2's LAMBADA accuracy (~45%) across all flex levels after fine-tuning
- **Bug**: Will crash at end of epoch 10 with `ZeroDivisionError` in `CosineAnnealingLR` (T_max=0). Training completes but **weights are lost** (temp checkpoint dir is deleted on crash). Must resubmit after pushing fixes.
- **Script**: `run_flexgpt_wikitext103_kd.sh`

### 2. `flexllama,fineweb.pretrained` — FlexLLaMA init from JackFram/llama-160m
- **Status**: Failed with same `ZeroDivisionError`. Must resubmit after pushing fixes.
- **Script**: `run_flexllama_fineweb_pretrained.sh`

### 3. `flexllama,fineweb.tiny` — 1-epoch sanity check
- **Status**: Completed successfully.

### 4. `flexgpt,wikitext103.kd_tiny` — 1-epoch KD sanity check
- **Status**: Not yet run (or recently submitted). Script: `run_flexgpt_kd_tiny.sh`

---

## Bugs Fixed (commit before resubmitting)

### Bug 1: ZeroDivisionError in CosineAnnealingLR (CRITICAL)
**File**: `config/experiments.py`

`GPTTrainingContext.make_scheduler` computes `T_max = epochs - warmup_epochs`. When both equal 10, `T_max=0` → division by zero in PyTorch's cosine scheduler. Crash happens after the final training epoch, before checkpoint is permanently saved → **trained weights are lost**.

**Fix applied**:
```python
# In GPTTrainingContext.make_scheduler:
cosine = CosineAnnealingLR(optimizer, T_max=max(1, self.epochs - self.warmup_epochs), eta_min=1e-5)
```
Also changed `warmup_epochs = 2` in both `FlexLMKDTrainingContext` and `LLaMATrainingContext` (was 10, same as their `epochs` default).

### Bug 2: Teacher weights saved to checkpoint (FIXED earlier)
`FlexLMKDTrainer` stores teacher via `object.__setattr__(self, '_teacher', ...)` to bypass PyTorch module registration so Lightning doesn't include teacher weights in checkpoints.

### Bug 3: Teacher is None during test-from-checkpoint (FIXED earlier)
`FlexLMKDTrainer` has `_ensure_teacher()` called from both `on_fit_start` and `on_test_start` to handle the case where Lightning loads a fresh instance from checkpoint for the test step.

### Bug 4: `load_fineweb_edu` not in safe_globals (FIXED earlier)
Added to `torch.serialization.add_safe_globals` in `training.py`.

---

## Key Files

| File | What changed |
|------|-------------|
| `config/experiments.py` | Added `FlexLMKDTrainingContext`, `LLaMATrainingContext`; added flexgpt KD experiments and flexllama experiments; fixed `T_max=max(1,...)` |
| `training.py` | Added `FlexLMKDTrainer` class; added `load_fineweb_edu` to safe_globals |
| `utils.py` | Added `load_openwebtext()`, `load_fineweb_edu()`, `load_llama_weights_into_flexllama()` |
| `flex_modules/rope_attention.py` | Custom RoPE attention for FlexLLaMA (fm.Linear projections, manual RoPE) |
| `networks/flexllama.py` | FlexLLaMA model and config |

---

## Pretrained Weight Loading

**FlexGPT**: `utils.load_gpt2_weights_into_flexgpt(model, "gpt2")`
- Called automatically by `FlexModelTrainer.handle_pretrained_hf()` when `FlexGPTConfig(pretrained_hf_model="gpt2")`
- NOTE: `pretrained_hf_model` in config is NOT used by `make_model()` directly — only by the trainer at `run_training()` time

**FlexLLaMA**: `utils.load_llama_weights_into_flexllama(model, "JackFram/llama-160m")`
- Same pattern, called automatically when `FlexLLaMAConfig(pretrained_hf_model="JackFram/llama-160m")`

---

## Snellius Setup

- Partition: `gpu_h100`
- Budget remaining: ~55,000 SBU (checked 2026-06-05)
- HF cache: `/scratch-shared/$USER/hf_cache` (scratch, may be cleaned)
- HF home: `/scratch-shared/$USER/hf_home`
- Env: `source ~/FlexViT/myenv/bin/activate`
- Logs: `~/FlexViT/logs/`

**After pushing fixes, resubmit**:
```bash
sbatch ~/FlexViT/run_flexgpt_wikitext103_kd.sh
sbatch ~/FlexViT/run_flexllama_fineweb_pretrained.sh
```

---

## KD Training Logic (FlexLMKDTrainer)

For each batch, per flex level:
```
L = kd_lambda * KL(student || teacher) + (1 - kd_lambda) * CE(student, true_tokens)
```
With `kd_lambda=1.0` (pure KL): dataset content is irrelevant, only teacher signal matters. WikiText-103 is used as the token stream but what the tokens say doesn't affect training.

Teacher: frozen GPT-2 Small (`gpt2`), loaded once per job, stored outside PyTorch module system.

---

## Known Risks

1. **Scratch-shared cleanup**: OpenWebText cache at `/scratch-shared/$USER/hf_cache` may be deleted by Snellius admins. The KD experiments use WikiText-103 (on home dir) so this doesn't affect them.
2. **Checkpoint temp dir**: The `finetune()` function uses `tempfile.TemporaryDirectory()`. If training crashes inside `trainer.fit()`, the temp checkpoint is deleted and weights are lost. The ZeroDivisionError fix prevents this.
3. **LAMBADA evaluation**: Run separately via `lm-evaluation-harness`. Not part of the training job — must be run after training completes.
