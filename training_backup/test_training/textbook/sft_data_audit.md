1.  think 得训，通常训练think的扩写，thinking message 扩写（summary），mask 会影响 distribution 

**2. Replay 被重复训练 —— 主要是 thinking，thinking block 也可以用 hash 统计一下**


**3. Thinking output 和 thinking replay 混在一起 —— 已处理，micro compact 去掉 think 的情况**


**4. Tool schema 缺失但 tool_call 被训练 —— 检查下有没有这种报错的幻觉情况和现在处理**


**5. Tool result replay / omitted result 问题 —— 已处理**


**6. Compression summary 是状态，不是普通 user content —— 已处理**


**7. System notifications 几乎全量存在 —— 可能导致分布不适合其他 harness**

**8. Actor attribution 风险 —— subagent 得再洗洗数据看有没有问题**

**9. Split 泄漏要按 session/thread —— 已经关注**


**需要检查是否修了的问题**
1. 重复
2. compact 和 tool result offload 修复？
3. think 标识处理
4. tool response（ms-swift），？？tool 中间要插入 assistant？？
5. **tool use 系统注入**
6. tool 幻觉问题
gpt热切claude的时候，不知道为啥gpt的trace也会被带进来训练

**还没处理**
1. category
2. 图片

**评测打通**：量、质量、多样性
先标 1k 条，prompt 角度清洗，一次性打完
- 人工看一下标的对不对；quality 打分；html 分数特别高，特别规整 —— category + 打分
- 分工之后散着看，所有 category 都看看并感受打分，人配合 agent 找问题
1. thinking 的训练
2. thread-id 分布处理
3. 质量打分
4. category 分布
