# Qwen3.6-27B Frappe/GLM three-run final report

Completed at 2026-08-07 04:19 UTC. All accepted runs used full-parameter SFT
on 16 H100 GPUs (2 nodes x 8 GPUs), TP4/PP2/CP2/DP1, micro batch 1, global
batch 2, two epochs, BF16, cosine LR from 1e-5 to 1e-6, max length 262144,
and CPU optimizer offload. No model was deployed and no manual heldout
evaluation was run.

## Accepted runs

| Arm | Dataset | Task | Steps | Loss first / mean / final | Peak GPU memory | Elapsed | Final checkpoint |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| A | 99 Frappe/GLM | `t-20260807044717-dc78r` | 99 | 0.18474187 / 0.17184717 / 0.11864355 | 51.26 GiB | 1h 23m 18s | `/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_frappe_glm_a99_16gpu_gbs2_epoch2_20260806_v3/v0-20260806-204944/checkpoint-99` |
| B | 47 SaaS no-skill + 99 Frappe/GLM | `t-20260807061117-sd8lx` | 146 | 0.53286195 / 0.17538117 / 0.13046010 | 50.27 GiB | 1h 47m 56s | `/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_saas47_frappe99_b146_16gpu_gbs2_epoch2_20260806_v3/v0-20260806-221343/checkpoint-146` |
| C | (47 SaaS + 99 Frappe/GLM) x2 + fixed-seed general 200 | `t-20260807080932-vdlmc` | 492 | 0.23737706 / 0.13818454 / 0.02629309 | 55.11 GiB | 4h 09m 04s | `/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_saas47_frappe99_x2_general200_c492_16gpu_gbs2_epoch2_20260806_v4/v0-20260807-001156/checkpoint-492` |

All three platform tasks ended in `Success`. Each final checkpoint has a valid
iteration marker, 12 non-empty safetensors shards, 1,184 indexed tensors,
54,713,457,120 indexed weight bytes, zero missing shards, two independent node
postflight confirmations, and zero fatal log matches.

## Dataset contract

| Arm | Rows | Unique trace IDs | Multiplicity | Trainable messages | Dataset SHA256 |
| --- | ---: | ---: | --- | ---: | --- |
| A | 99 | 99 | 99 x1 | 8,999 | `4c315dcebf9e2b00bcab14075478852bddcc1077fabadfb555f52f1170101b44` |
| B | 146 | 146 | 146 x1 | 11,620 | `8cb425c7cb234d43292f6b439d38576fe50295bdfa3757794300b196816e2d93` |
| C | 492 | 346 | 146 x2 + 200 x1 | 36,656 | `851ab2a30f50b4b5e5aea4fdba1639acf626afdc5f8343daafd2672f77fba425` |

The general 200 rows were selected with fixed seed `20260806` after excluding
the 99 Frappe/GLM and 47 SaaS trace IDs. Four overlength rows from the initial
reservoir sample were deterministically replaced by four validated rows before
training. Exact Qwen3.6 template rendering passed for every accepted row.

## Arm C retry record

The first Arm C attempt, `t-20260807075948-t4jss`, was cancelled after
HuggingFace/PyArrow rejected mixed integer/float values in the non-training
metadata field `dedup_stats.trainable_turn_ratio`. The v4 dataset converts only
34 integer metadata values to numerically equivalent floats. Messages, tools,
loss masks, trace IDs, ordering, multiplicities, and trainable targets are
unchanged. A fresh HuggingFace/PyArrow load of all 492 rows passed before the
accepted retry was submitted.

## Verification evidence

- `logs/qwen36_three_run_final_checkpoint_audit.json`: fresh platform and
  checkpoint verification for all accepted tasks.
- `logs/qwen36_three_run_final_preflight_postflight_audit.txt`: dataset SHA,
  row/multiplicity counts, trainable-message counts, and node postflight data.
- `logs/qwen36_arm_c_arrow_type_red.txt`: reproduces the C v3 type drift.
- `logs/qwen36_arm_c_arrow_type_green.txt`: proves the C v4 scalar type contract.
- `logs/qwen36_arm_c_hf_load_green.txt`: proves the full 492-row HF/PyArrow load
  and row-by-row semantic equivalence.
- `data/qwen36_frappe_glm_three_runs_20260806_v3/dataset_manifest.json`: A/B and
  parent C lineage.
- `data/qwen36_frappe_glm_three_runs_20260806_v4/arm_c_arrow_type_manifest.json`:
  C v4 parent/output hashes and transformation record.
