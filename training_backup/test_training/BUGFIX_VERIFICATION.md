# Arm C Arrow type normalization verification

## Bug

Arm C v3 mixed integer and floating-point values at
`dedup_stats.trainable_turn_ratio`. HuggingFace Datasets inferred an `int64`
Arrow field from an early JSON chunk and then failed on a later float before
training began.

## RED evidence

- Failed task: `t-20260807075948-t4jss`.
- Runtime error: `pyarrow.lib.ArrowInvalid: Float value 0.039046 was truncated converting to int64`.
- `logs/qwen36_arm_c_arrow_type_red.txt` records 34 integers and 60 floats at
  the same scalar path and exits non-zero.
- `logs/qwen36_arm_c_retry_assets_red.txt` proves the retry asset contract
  failed before the v4 task configuration existed.

## GREEN evidence

- Arm C v4 changes only 34 integer metadata values to numerically equivalent
  floats; it does not modify messages, tools, loss masks, trace IDs, ordering,
  or any trainable target.
- `logs/qwen36_arm_c_arrow_type_green.txt` reports zero mixed numeric paths for
  all 492 rows.
- `logs/qwen36_arm_c_hf_load_green.txt` records a fresh HuggingFace/PyArrow
  load of all 492 rows and a row-by-row semantic-equivalence check.
- `logs/qwen36_arm_c_retry_assets_green.txt` records a passing static check of
  the v4 SHA, PFS paths, task name, expected row/multiplicity counts, and the
  checkpoint-gated controller.

## Clean-state gate

The failed v3 task is retained in chain-state history. The retry uses a new v4
dataset hash, task name, and output directory, so failed logs or partial output
cannot be mistaken for the accepted checkpoint. The v4 task
`t-20260807080932-vdlmc` reached Success. `checkpoint-492` passed the remote
shard/log/postflight verifier with 12 non-empty safetensors shards, 1,184
indexed tensors, two node postflight confirmations, and zero fatal log matches.
The retry therefore finishes in a verified clean state.
