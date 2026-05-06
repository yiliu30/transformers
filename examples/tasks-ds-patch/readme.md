# DeepSeek V4 standalone downstream patches

This directory keeps the DeepSeek V4 workaround out of `src/transformers`.

## Files
- `patches.py`: standalone runtime patches for DeepSeek V4 experts and FP8 grouped linear loading
- `ds_v4.py`: example runner that opts into those patches
- `clean-code.md`: original task note

## What gets patched
- `DeepseekV4Experts` is replaced at runtime with a standalone expert block that keeps the checkpoint's `w1` / `w2` / `w3` layout.
- `DeepseekV4GroupedLinear` is replaced at runtime with a version that can keep its grouped forward while accepting prequantized FP8 weights.
- `replace_with_fp8_linear(...)` is monkey-patched only inside the example's context so DeepSeek V4 uses the standalone modules without touching core source files.

## Run
```bash
/home/yiliu7/workspace/venvs/dsv4-tran/bin/python examples/tasks-ds-patch/ds_v4.py
```

Optional flags:
```bash
/home/yiliu7/workspace/venvs/dsv4-tran/bin/python examples/tasks-ds-patch/ds_v4.py \
  --model /mnt/disk1/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
  --prompt "What is deep learning?" \
  --max-new-tokens 20
```

## Notes
- The patches are opt-in and scoped to the helper context in `ds_v4.py`.
- Core `transformers` source stays reverted to the upstream implementation.
