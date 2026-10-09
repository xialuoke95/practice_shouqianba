# Loss Alignment, Data Cleaning, And Mask Lessons

Last checked: 2026-06-09 UTC.

## Main Result

The large loss mismatch was primarily caused by data rendering and target-mask
semantics, especially around `tool_call`.

Before mask alignment:

- Local Bridge chat-JSONL eval: `lm loss=2.729141`, `PPL=15.31973`.
- Other platform logged step-1 training loss: `0.64677322`.

After reproducing the SWIFT/PFS mask semantics with a packed Bridge dataset:

- Local exact-mask packed Bridge eval: `lm loss=0.6624455`, `PPL=1.93953`.
- Difference from other-platform step 1: `0.01567228`, about `2.4%` relative.

This strongly indicates that the original `2.7` versus `0.65` gap was not
mainly a Megatron runtime numerical issue. It was dominated by the data and mask
surface being different.

## Evidence Roots

Local/West-US evidence:

- `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/`
- `s3://verdent.tech/test_training/loss_alignment_full_repro/qwen36-cleanv5-align-20260603/`
- `s3://verdent.tech/test_training/qwen36_cp_alignment/qwen36-cp-alignment-20260531-143057/`

Other-platform/TOS evidence:

- `s3://verdent-tos-public/public/task_test_trajs/full_repro_20260603/`
- The TOS root is not anonymously listable in the current environment. The
  report evidence says signed TOS credentials were needed.

The related B300 overlay alignment run is:

- `s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/`
- B300 manifest:
  `artifacts/source/k8s/21-train-swift-job.yaml`
- Kubernetes Job: `qwen36-mb-swift-aligned-a11`
- Node selector: `ml.p6-b300.48xlarge`
- Dataset key: `sft_opus_sonnet_swift_mbjson_v2`
- Loss/topology flags: `calculate_per_token_loss=true`,
  `ddp_average_in_collective=false`, `TP=2`, `PP=1`, `CP=2`, `EP=8`,
  `seq_length=128000`

Downloaded local evidence:

- `/tmp/test_training_evidence/loss_alignment_update_20260603.md`
- `/tmp/test_training_evidence/mb_comparison_report.md`
- `/tmp/test_training_evidence/cp_alignment_report.md`
- `/tmp/test_training_evidence/mask_snapshots_summary.json`
- `/tmp/test_training_evidence/cleanv5_baseline_status_completed.json`

## Data-Cleaning Findings

The raw data changed materially between the local run and the other-platform
cleaned run.

Local original raw data:

- File:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/data/raw/sft_opus_sonnet_swift_noimage.jsonl`
- Rows: 16,620.
- SHA256:
  `ee065acea4a7741cf61f454d398e2f673fb4d4dc37a5bdc058b739d61e41b676`.
- Bridge-prepared kept rows: 16,551.

Other-platform `clean_v5`:

- File name: `sft_opus_sonnet_swift_noimage_clean_v5.jsonl`.
- Rows: 13,668.
- Size: 2,070,934,059 bytes.
- SHA256:
  `f8dbe0dc24f5ba73700652d450c09b8802e8ec6eb7910d4a2fa3cd2e321074b0`.

Alignment audit:

- Matched by id/trace: 13,668.
- Canonical JSON identical: 1,828.
- Canonical JSON different: 11,840.
- Messages identical: 10,867.
- Messages different: 2,801.
- Top-level `tools` added/present: 11,840.

Lesson:

- Matching row ids is not enough. Preserve and compare canonical JSON, message
  roles, top-level `tools`, rendered text hashes, token hashes, label hashes,
  and loss-mask hashes.
- Keep the cleaning script/version with the dataset. A cleaned JSONL checksum is
  necessary but does not explain how the training surface was produced.

## Mask Semantics

Other-platform rendering/mask method:

- Render with Hugging Face `AutoTokenizer.apply_chat_template`.
- Normalize roles before rendering:
  - `tool_call -> assistant.tool_calls`
  - `tool_response -> tool`
- Train assistant text and trainable `tool_call` payloads.
- Exclude messages marked `loss=0`.

Sampled mask counts from the other-platform package:

- Sample count: 20.
- Trainable source counts:
  - `assistant=341`
  - `tool_call=584`
- `loss=0` masked counts:
  - `assistant=147`
  - `tool_call=233`
- `max_length=128000`.
- No sampled row was truncated at 128K.
- Maximum sampled token count after truncation: 75,761.

Original local Bridge prep policy:

- Trained non-masked assistant messages.
- Folded `tool_call`, `tool_response`, and masked assistant/tool-call messages
  into context.

Lesson:

- Tool-call role handling is part of the label definition. If one platform
  trains tool-call payloads and the other treats them as context, the loss
  denominator and target distribution are different.
- Store mask snapshots as first-class artifacts. The useful snapshot fields are
  rendered text hash, input token hash, label hash, loss-mask hash, target token
  counts, and per-role target counts.

## Volcano/TOS/PFS Alignment Chain

The cross-platform evidence chain uses several names for the non-local side:
TOS public artifacts, PFS paths inside the training platform, and the
other-platform/SWIFT Megatron run. In these notes they refer to the same loss
alignment target, not separate baselines.

The important chain was:

- TOS package exposes `clean_v5`, PFS run args/logging, source artifacts, base
  model, and `checkpoint-854` metadata.
- PFS run args establish `TP=2`, `PP=2`, `CP=2`, `EP=4`,
  `seq_length=128000`, `global_batch_size=32`, and
  `calculate_per_token_loss=true`.
- PFS logs establish step-1 training loss `0.64677322`.
- Source artifact mask snapshots establish the SWIFT rendering and mask
  semantics, especially trainable `tool_call` payloads.
- The local exact-mask packed Bridge reproduction reached `lm loss=0.6624455`,
  showing the old `2.729141` loss was caused mainly by data/mask mismatch.

Operational lesson:

- Do not compare loss scalars until the raw data checksum, role normalization,
  chat template, target mask, truncation policy, per-token loss mode, and
  rank-local numerator/denominator semantics are all pinned.
- A cleaned JSONL alone is not enough; the PFS/SWIFT mask construction is part
  of the training data.

## Packed Dataset Reproduction

The exact SWIFT-mask packed Bridge baseline used:

- Script: `prepare_swift_exact_packed_npy.py`.
- S3 packed split:
  `s3://verdent.tech/test_training/loss_alignment_full_repro/qwen36-cleanv5-align-20260603/swift_exact_packed/first64_left16k_seqstartlist/`.
- HF `apply_chat_template` after role normalization.
- Assistant and trainable tool-call targets.
- Assistant headers and `<|im_end|>` excluded from loss.
- Source loss mask shifted by one position for Bridge packed labels.
- `seq_start_id` stored as a Python list, not a numpy array.

Bridge attempt history:

- r1 failed because Bridge expected default packed metadata at
  `packed/Qwen--Qwen3.6-35B-A3B_pad_seq_to_mult1/16384_metadata.jsonl`.
  Adding an empty `[]` metadata file fixed it.
- r2 failed because `seq_start_id` as a numpy array broke Bridge 0.4.1 packed
  collate. Regenerating it as a Python list fixed it.
- r3 completed with `lm loss=0.6624455`, `PPL=1.93953`.

Lesson:

- For packed Bridge datasets, keep tiny metadata and structural fields exactly
  compatible with Bridge's collate assumptions.
- A 20-sample hash match is a good smoke test, but it is not a proof of full
  numerator/denominator equality.

## Training Config Differences

Other-platform run:

- Framework entrypoint: `swift.cli._megatron.sft`.
- Platform task type: PyTorchDDP.
- Image: `pytorch:2.7.1-cu12.8.1-py3.12-ubuntu22.04`.
- Topology: `TP=2`, `PP=2`, `CP=2`, `EP=4`.
- 4 nodes x 8 H100.
- `seq_length=128000`.
- `global_batch_size=32`, `micro_batch_size=1`.
- `calculate_per_token_loss=true`.
- `padding_free=true`.
- LR `1e-5`, min LR `1e-6`.
- `moe_aux_loss_coeff=0`.

Local Bridge baseline:

- Image: AWS DLC PyTorch `2.8.0-gpu-py312-cu129`.
- Megatron-Bridge `0.4.1`, Megatron-Core `0.17.0rc0`.
- Topology: `TP=2`, `PP=1`, `CP=1`, `EP=8`.
- 1 node x 8 H200.
- `seq_length=16384`.
- `global_batch_size=16` for the old all-data run, `32` for the exact-mask
  first64 packed run.
- `calculate_per_token_loss=true` requires
  `ddp.average_in_collective=false`.

Lesson:

- `calculate_per_token_loss=true` cannot be combined with
  `average_in_collective=true` in this Megatron-Bridge path.
- Once `average_in_collective=false` is set, per-token loss alone did not fix
  the old chat-JSONL loss. Mask/rendering was the dominant factor.

## CP Alignment Lessons

The CP alignment probe compared CP=1 and CP=2 behavior on qwen_text runs.

Completed probes:

- `qwen36_cp_align_cp1_legacy_all_r4`:
  train loss `2.02743 -> 1.75083`, reload NLL `1.93546`.
- `qwen36_cp_align_cp2_lr15_all_r4`:
  train loss `2.02916 -> 1.5124`, reload NLL `1.7427`.
- `qwen36_cp_align_cp1_legacy_split_r4`:
  reload train/test NLL `2.044 / 1.56369`.
- `qwen36_cp_align_cp2_lr15_split_r4`:
  reload train/test NLL `1.88118 / 1.40014`.

Failed/incomplete probes included combinations around CP=2, HF template, and
per-token/average-collective settings.

Lesson:

- CP changes can alter effective update scale and loss interpretation.
- Higher LR under CP=2 improved the train/reload delta in the probe, but did not
  by itself make the run CP=1-like.
- Keep CP, LR, loss reduction mode, and dataset/mask path together in every
  comparison.

## Data Governance Checklist

For future loss-alignment work, archive these as mandatory artifacts:

- Raw JSONL checksum, size, row count, and version id.
- Cleaning script checksum and parameters.
- Canonicalized post-clean JSON checksum.
- Role-normalization script checksum.
- Chat-template checksum.
- Tokenizer files and tokenizer metadata checksum.
- Rendered-text hash samples.
- Input token hash samples.
- Label hash samples.
- Loss-mask hash samples.
- Per-role trainable token/source counts.
- Truncation statistics by split.
- Sampler order or first-step sample ids.
- Rank-local loss numerator/token denominator for the first few steps.
- Full training config, including TP/PP/CP/EP and loss-reduction flags.
