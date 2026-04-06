# Stage 1 向 CLISA 靠拢的改造计划

## 0. 前提：两个数据集的根本差异

在讨论改造之前，必须先理解数据集的差异——这决定了哪些改造是合理的，哪些是勉强的。

| | CLISA 的数据 (SRT/情绪) | 当前数据 (EEGMAT/认知负荷) |
|--|------------------------|--------------------------|
| 刺激形式 | 28 段视频，所有被试看**同一个**视频 | 口头给出数字做连续减法，每人**不同的数字** |
| 条件数 | 28 个 video (或 9/3/2 类情绪) | 2 个条件（休息/心算） |
| 每条件时长 | 30s × 28 = 840s | 60s × 1 = 60s |
| 刺激对齐性 | **高**：同一 video 同一时刻，所有被试看到相同画面 | **低**：休息无结构，心算各人不同时刻做不同题 |
| 每被试 epoch 数 | 28 × 13 = 364 | ~30 × 2 = 60 |
| 被试数 | 123 | 36 |

---

## 1. 改造计划：可行的修改项

### 1.1 Loss: SupConLoss → InfoNCE

**改动**: 将 `losses.py` 中 Stage 1 使用的 `SupConLoss` 替换为已有的 `InfoNCELoss`。

**做法**: 在 `Stage1Trainer` 中，不再把所有同 label 的样本作为正样本，而是只把 `(z_a[i], z_b[i])` 这一对作为正样本。

**效果**: 每个 anchor 从 ~63 个正样本变为 1 个正样本，对比信号更集中。

**可行性**: ★★★★★ — 代码改动最小，`InfoNCELoss` 已经存在，只需改调用方式。

```python
# 现在
features = cat([z_a, z_b])
all_labels = cat([labels, labels])
loss, acc = SupConLoss(features, all_labels)

# 改为
loss, acc = InfoNCELoss(z_a, z_b)
```

### 1.2 采样: 按 label 均匀 → 按 epoch 位置对齐

**改动**: 修改 `CrossSubjectPairDataset`，不再按 label 采样，而是让被试 A 的第 k 个 epoch 和被试 B 的第 k 个 epoch 配对。

**做法**:
- 为每个被试的每个条件内的 epoch 按时间顺序编号
- 采样时，对被试对 (A, B) 取相同编号的 epoch

**可行性**: ★★★☆☆ — 需要在数据加载时保留 epoch 的时间顺序索引。

### 1.3 归一化: BatchNorm → Stratified LayerNorm

**改动**: 在 `EEGEncoder` 中增加可选的 Stratified LayerNorm，训练时按被试分组归一化。

**做法**:
- 已有 `StratifiedBatchNorm` 类（在 `encoder.py` 第 42-65 行），但目前 encoder forward 中未使用
- 可在 temporal/spatial block 之间插入 stratified 归一化

**可行性**: ★★★★☆ — 代码骨架已存在，需要修改 forward 传入 subject_ids。

### 1.4 Projection Head: 保留 or 移除

**改动**: CLISA 没有独立的 projection head，直接用卷积展平的特征做对比。

**建议**: 保留 projection head。这是 SimCLR 论文验证过的设计——projection head 保护 encoder 的表征质量。移除它是 CLISA 的简化设计，不是优势。

**可行性**: 不建议改动。

### 1.5 去掉标签依赖 → 自监督

**改动**: 预训练阶段不使用 labels，完全依赖被试对+位置对齐作为自监督信号。

**可行性**: ★★☆☆☆ — 见下方"不自然之处"第1条。

### 1.6 卷积顺序: 时间→空间 改为 空间→时间

**改动**: 改为 CLISA 的顺序。

**建议**: 不建议改动。时间→空间是 EEGNet / ShallowConvNet 验证过的有效顺序，且当前 encoder 已适配 eegmat 的通道数和时间长度。

---

## 2. 不自然之处（需要审阅的风险点）

### ⚠️ 1. 核心矛盾：EEGMAT 没有"共享刺激"来提供对齐信号

这是**最根本的问题**。

CLISA 的自监督信号来源：
> 被试 A 在 t=15s 看视频 #3 → 看到"一个人在笑"
> 被试 B 在 t=15s 看视频 #3 → 看到同一个画面"一个人在笑"
> → 大脑响应应该相似 → 正样本

EEGMAT 的情况：
> 被试 A 在 t=15s → 休息，可能在走神
> 被试 B 在 t=15s → 休息，可能在想午饭
> → 大脑响应**没有理由相似**

> 被试 A 在 t=15s → 在算 3141 - 42 = 3099
> 被试 B 在 t=15s → 在算 2758 - 37 = 2721
> → 认知负荷状态相似，但**具体神经活动模式不同**

**结论**: 如果改成 CLISA 式的"同位置配对自监督"，正样本的假设不成立。模型可能学到的是与认知状态无关的伪信号（如录制环境噪声的时间特性、电极漂移的时间趋势等）。

### ⚠️ 2. "Stimulus Identity" 数量太少

CLISA 的 batch 有 28 个不同的 video identity，提供了丰富的对比信号（1 正 vs 54 负）。

EEGMAT 只有 2 个条件。如果把 epoch 位置作为 identity:
- 每个条件 ~30 个 epoch → 一个 batch 最多 30 个 identity
- 但这 30 个 epoch 之间并没有像 28 个 video 那样具有本质上不同的刺激内容
- 前后 epoch 之间高度相关（相邻 2 秒的脑电非常相似）
- **负样本的区分度不够**：epoch 5 和 epoch 6 的脑电差异可能很小

### ⚠️ 3. 数据量对比悬殊

| | CLISA | EEGMAT |
|--|-------|--------|
| 被试对数 | C(110,2) ≈ 5995 | C(~24,2) ≈ 276 |
| 每对的batch大小 | 56 | 需要重新设计 |
| 总训练iteration/epoch | 5995 | 276 |

如果模仿 CLISA 的"每对被试一个 batch"策略，EEGMAT 的训练量会少一个数量级。需要增加 `n_times`（每对重复采样次数）来补偿。

### ⚠️ 4. Stratified LayerNorm 的适用性

CLISA 的 Stratified LayerNorm 设计用于处理不同被试的 EEG 幅度差异。这在 EEGMAT 上也是合理的——但当前代码已经在数据预处理阶段做了 z-score 归一化。

**双重归一化（预处理 z-score + 模型内 stratified norm）可能过度归一化**，丢失有用的幅度信息。需要二选一，或者验证是否互补。

### ⚠️ 5. 去掉标签是"退步"而非"进步"

CLISA 不用标签是因为它有更好的自监督信号（刺激对齐）。EEGMAT **缺乏这种信号**，此时：
- 用标签做有监督对比 → 信号质量高、方向明确
- 不用标签做自监督对比 → 对齐信号弱、可能学到噪声

**在缺乏强自监督信号的数据上，放弃标签没有理论或实践上的好处。**

### ⚠️ 6. epoch 位置对齐的隐含假设

如果用"同一时间位置的 epoch"做正对，隐含假设是：
> 所有被试在录制第 k 个 epoch 时处于相同的认知子状态

这对于 60 秒的持续任务来说不成立——有人在第 10 秒就进入状态，有人在第 30 秒才进入状态，疲劳效应也因人而异。

---

## 3. 推荐方案：选择性靠拢

基于以上分析，推荐**保留标签信息**的前提下，选择性吸收 CLISA 的优点：

| 改动项 | 是否采纳 | 理由 |
|--------|---------|------|
| Loss: SupConLoss → InfoNCE | ✅ 建议尝试 | 减少同条件不同内容的样本被拉到一起的问题；保留 SupConLoss 作为对照 |
| 采样: 加入 epoch 位置约束 | ⚠️ 可选实验 | 在相同条件、相同位置下配对，兼具 label 和位置信息 |
| Stratified LayerNorm | ✅ 建议加入 | 消除被试间幅度差异，但需要与预处理 z-score 协调 |
| 去掉标签依赖 | ❌ 不建议 | EEGMAT 无共享刺激，去掉标签会失去最可靠的对比信号 |
| 去掉 Projection Head | ❌ 不建议 | Projection head 经过验证是有益的 |
| 改卷积顺序 | ❌ 不建议 | 当前顺序已验证有效，无需跟随 CLISA |

### 具体改动计划（按优先级排序）

1. **在 `Stage1Trainer` 中增加 loss 切换选项** — 支持 SupConLoss 和 InfoNCE 两种模式
2. **激活 `StratifiedBatchNorm`** — 修改 `EEGEncoder.forward()` 接收 `subject_ids` 参数
3. **采样策略增加"条件内位置约束"选项** — 在 `CrossSubjectPairDataset` 中，同 label 配对时优先选择相近时间位置的 epoch
4. **添加 ablation 实验配置** — 便于对比 SupCon vs InfoNCE、有无 Stratified Norm 的效果

---

## 4. 总结

CLISA 的核心创新是利用**共享刺激的时间对齐**作为自监督信号，这在视频观看型实验中非常自然。但 EEGMAT（连续减法任务）**不具备这一条件**。因此，完全照搬 CLISA 会引入没有物理意义的假设。

推荐的策略是**"取其形、留其实"**：
- 借鉴 CLISA 的技术手段（InfoNCE、Stratified Norm）
- 保留当前数据集适合的监督信号（label）
- 不强行模仿没有数据基础的自监督对齐
