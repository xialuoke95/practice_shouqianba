数据
- 电商：tau-bench
- 基础数据：## **Databricks Dolly-15k**

agent 给的结论：
1) 大致没错，但理解并不深入，逻辑推理需要一步步引导
2) 没有一个明确能成功的任务作为 reward 上的指引
dolly probe 则意义过于缺失，线上任务则难度过大
想清楚分布的问题到底在哪里

## recipe 本身怎么学
![[attachments/Pasted image 20260524015301.png]]


## 学习顺序
![[attachments/Pasted image 20260514013843.png]]
![[attachments/Pasted image 20260514014702.png]]

实验顺序：
1. eval
2. LoRA 训练
3. 分布式演练
4. scale 版本

## 学习方向

![[attachments/fff164372e99a9507c1d9c307c51fe3d.png]]


Brazil Localization SFT / Post-training
├── 算法侧
│   ├── 训练目标与数据
│   │   ├── SFT objective / cross entropy / label mask
│   │   ├── **chat template** / tool-call serialization / packing
│   │   └── localization 数据切片：风格、流程、格式、工具、本地知识
│   │
│   ├── 参数高效训练
│   │   ├── LoRA：**target module**、rank、alpha、LR
│   │   ├── QLoRA：4-bit base、NF4、量化噪声、公平对比
│   │   └── PTQ/QAT：部署相关，第一阶段不做主线
│   │
│   └── MoE / MLA / Kimi 迁移
│       ├── MoE：router、expert load、shared/routed experts
│       ├── MLA：attention projection 不再等同于 q/k/v/o
│       └── 迁移原则：迁移审计流程，不迁移模块名和超参
│
└── 工程侧
    ├── **分布式与显存**
    │   ├── DP / ZeRO / FSDP / TP / EP / PP / SP
    │   ├── 参数、梯度、优化器、activation 显存结构
    │   └── tokens/sec、step time、GPU memory、expert imbalance
    │
    ├── 框架与生命周期
    │   ├── LLaMA-Factory：快速 baseline
    │   ├── ms-swift / Megatron-SWIFT：主线 SFT / MoE 多机训练
    │   └── checkpoint、resume、export、merge、inference
    │
    └── 观测与审核
        ├── run card、data hash、model revision、env version
        ├── raw generation、slice eval、failure taxonomy
        └── code agent 审核：loss mask、target modules、可比性


## 实验矩阵
![[attachments/Pasted image 20260514011633.png]]

### Artifacts
![[Pasted image 20260514014319.png]]

## OPD

![[attachments/Pasted image 20260520120110.png]]

![[attachments/Pasted image 20260520121110.png]]


hint 不能去学到过度自信的部分，不能直接看答案

## Lessons: swe-spot 经验总结

整个 trajectory 也是能训的，根据 tool_result 可以训下一步适合调用什么工具

### MOE & MLA

LoRA merge & export

MoE 需要关注：

- router。

- expert load。

- shared experts。

- routed experts。

- expert imbalance。

- EP 与 DP/TP 的组合。


MLA 需要关注：

- attention projection 不再等同于传统 q/k/v/o。

- target module 需要按模型结构审计。

- 不应把 dense attention 的 LoRA target 机械迁移过来。


### 工程
MFU

需要理解的并行方式：
- DP：data parallel。

- ZeRO：optimizer/gradient/parameter sharding。

- FSDP：fully sharded data parallel。

- TP：tensor parallel。

- EP：expert parallel。

- PP：pipeline parallel。

- SP：sequence parallel。

  

显存结构与显存计算：
- 参数。

- 梯度。

- 优化器状态。

- activation。

- KV/cache 或训练中间状态。

- communication buffer。


## 训练知识-快速

Advisor模式：得说少量的看看

- benchmark
    
- case
    

直观理解看看

复述 hint，得理解 + 作出新的行动

涨了 performance，hint 去掉也自然，才能训

  
Seed + opus hint，最开始没调好 + 不能很明确指出答案，涨幅不明显，需要细调

有一个自己的很轴的方向

Judge 是不是蒸馏：隐晦

1w 条和 500 条，和不蒸馏，是不一样的 pattern。分数差不多，pattern 差很多

- Traj pattern 无法压下去
    

  

下一批数据，看看 reward 是否靠谱

不复述：

- 看到 advisor 不回答
    
- Advisor 把答案告诉他，等于抄，也学不进去
    
- 看到 advisor 后内化到模型自己的认知，重新复述 —— 需要的数据 —— 需要调
    

如果是 hook 进去，占比不高

少了 token 直接训

Kimi 1T -> int4 -> 转 fp8 训练，训练是否稳定需要基建

20B -> 200B -> 1T

线上洗数据 + 蒸 + 看 loss -> 没有评测

- 看 PPL
- 看 case 打分

线上拉数据看 pattern

硬蒸效果最快，先铺路


数据
- 质量
- 多样性
- 数量

1条数据可能有很多 session
10%+用户觉得好，10%+用户觉得糟；旁路没有这部分信号




1. 优质数据 + GPT 的数据，还是 claude 的数据？
	1. 洗掉很大一部分
2. 40w * 3w token/session = 12B，128卡，fp8，半天一轮

既要有我们的环境，也要有公开数据

**7月之前有结果？**
高质量用户数据


1. 洗数据 + 强行试一试
	1. loss 和 PPL
2. SWE-bench on kimi 先测测 50 题
	1. 训完的也要想办法看看怎么测 50 题，需要把模型 serving 起来
	2. kimi 2.6 的自部署

> 训练可用性评估

没有部署，是怎么测起来的？

原生 kimi < 我们的 harness。builder 不垂

Qwen30A3
Qwen3-235B-A22B
sft 的长度分布，数据质量与多样性
只有 trajectory，也能训 sft，tau-bench

看看2机16卡最大能训起来多少

模型：Qwen3-235B-A22B-Instruct-2507  
框架：Megatron-SWIFT  
方法：LoRA SFT  
精度：BF16 起步，FP8 后置  
seq_len：4096 or 8192 起步  
micro_batch_size：1  
global batch：靠 grad_accum 堆  
tokens：先 100M，再 1B，再 10B  
保存：每 200–500 steps  
eval：每个 checkpoint 跑 Brazil localization eval

attention projection only

GPT-traj 有多少

**数据形式不能让模型找到规律，关键信息在 prompt 中的位置分布要足够随机**

评测手段

**今天先跑通2机16卡**，235B

## From 苏老师


1.  think 得训，通常训练think的扩写，thinking message 扩写（summary），mask 会影响 distribution 

**2. Replay 被重复训练 —— 主要是 thinking，thinking block 也可以用 hash 统计一下**


**3. Thinking output 和 thinking replay 混在一起 —— 已处理，micro compact 去掉 think 的情况**


**4. Tool schema 缺失但 tool_call 被训练 —— 检查下有没有这种报错的幻觉情况和现在处理**


**5. Tool result replay / omitted result 问题 —— 已处理**


**6. Compression summary 是状态，不是普通 user content —— 已处理**


**7. System notifications 几乎全量存在 —— 可能导致分布不适合其他 harness**

**8. Actor attribution 风险 —— subagent 得再洗洗数据看有没有问题**

**9. Split 泄漏要按 session/thread —— 已经关注**

**已经修了的问题**
1. 重复
2. compact 和 tool result offload 修复？
3. think 标识处理
4. tool response（ms-swift），？？tool 中间要插入 assistant？？
5. **tool use 系统注入**

**还没处理**
1. category
2. 图片

**评测打通**：量、质量、多样性
合并数据 pipeline skill
先标 1k 条，prompt 角度清洗，一次性打完
- 人工看一下标的对不对；quality 打分；html 分数特别高，特别规整 —— category + 打分
- 分工之后散着看，所有 category 都看看并感受打分，人配合 agent 找问题
1. thinking 的训练
2. thread-id 分布处理
3. 质量打分
4. category 分布
  
swe-bench evaluation，掉点，aws + 火山云

CP=2, B300, 8小时，1.6w ，8k，global-batch=16，MFU；128k ，chunk 
32卡 H200，epoch = 2, 10h

**thinking 怎么训练**
Coconut、Huginn、Quiet-STaR、Pause Tokens、Implicit CoT、Stepwise Internalization、STaR、LaTRO、Searchformer、Looped Transformer、SemCoT、Soft Thinking