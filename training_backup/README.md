# 训练实验目录备份

2026-10-09 将本机两个实验目录合并备份到此处，保留各自目录结构：

- `test_training/`：训练与部署脚本、BytePlus / HyperPod 配置、运行说明、实验总结及结构化验收记录。
- `fireworks/data_alignment/`：数据转换、renderer 适配、训练入口、配置及汇总报告。
- `backup/`：原文件清单、外部数据与 checkpoint 的来源记录、备份说明及 SHA-256 校验表。

## 备份范围

本次保存 266 个原文件，合计 1,343,418 字节。它们来自 `/home/index/workspace/test_training/` 和 `/home/index/workspace/fireworks/data_alignment/`，导出过程没有修改源目录。每个复制文件均核对了源文件与备份文件的 SHA-256。

本仓库公开，因此没有上传原始训练 JSONL、样本级审计流、运行日志、W&B 历史和可能内嵌原始样本的 HTML。另排除本地 HyperPod 状态、通知标记及运行符号链接，共 72 项；这些项目的来源路径、大小、排除原因和可读取文件的 SHA-256 保存在 `backup/source_inventory.json`。Git 元数据、缓存和虚拟环境不属于导出范围。

现有脚本依赖环境变量或本机凭据配置。复制文件经过凭据扫描，没有发现需替换的凭据；运行时仍须自行提供合法凭据。此备份没有创建凭据，也没有执行训练、部署、评测或旧监控脚本。

## 数据和 checkpoint

数据的体积和校验值可通过 `backup/source_inventory.json` 核对。已有实验清单、数据版本来源与历史校验值保留在 `test_training/data/`；外部数据及 checkpoint 的结构化来源索引见 `backup/external_assets.json`。配置与报告中其他存储路径也保留原文。

PFS / S3 上的数据和模型权重不在 Git 中。本次没有访问这些外部存储，也没有重新计算 checkpoint 校验值。记录中“训练完成”“分片完整”等结论来自原实验记录，只说明原记录的判断，不代表本次恢复或训练验收。恢复权重时应在对应存储侧重新核对文件和分片校验值。

## 核验与恢复

从仓库克隆后，在仓库根目录运行：

```bash
cd training_backup
sha256sum -c backup/files.sha256
```

`backup/snapshot.json` 记录导出时间、来源、排除范围与检查结果。Python、Shell、JSON 和 YAML 已做语法或格式检查；这些检查不证明当前依赖、GPU 或云端权限可以运行全部历史实验。

需要恢复某个原目录时，复制相应目录，再从来源存储补齐清单中排除的数据、日志或模型文件。`backup_manifest_kimi_glm_20260721.sha256` 是原历史清单，本次备份的统一校验入口是 `backup/files.sha256`。
