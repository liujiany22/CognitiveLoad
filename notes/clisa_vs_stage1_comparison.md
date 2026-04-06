# CLISA vs DualAlign Stage 1 详细对比

## 概览

| 维度 | CLISA (`Clisa_analysis/`) | DualAlign Stage 1 (`code/`) |
|------|--------------------------|----------------------------|
| **学习范式** | 自监督对比学习（不需要标签） | 有监督对比学习（需要标签） |
| **Loss** | InfoNCE (NT-Xent) | SupConLoss (Khosla 2020) |
| **正样本定义** | 同 video + 同 segment + 不同被试 | 同 label + 不同被试 |
| **正样本数量/anchor** | 1 | 多个（所有同 label 的样本） |
| **采样单元** | 被试对 → 共享 video segment 索引 | 被试对 → 按 label 均匀采样 |
| **模型** | ConvNet + Stratified LayerNorm | EEGEncoder (ChannelAttn + BN) + MLP Projector |
| **优化器** | Adam + CosineAnnealingWarmRestarts | AdamW + CosineAnnealingLR + Grad Clip |
| **Early Stopping** | 基于 val accuracy | 基于 val loss |

---

## 1. 数据加载与预处理

### CLISA

```
数据来源: 情绪识别 EEG 数据
格式:     pickle 文件 → (n_subs, 28 videos, 30 channels, 7500 points)
采样率:   250 Hz
视频时长: 30 秒
滑窗切段: timeLen=5s, timeStep=2s → 13 segments/video
每被试:   28 × 13 = 364 segments
```

- 数据 reshape 成 `(n_subs, 364*n_channs)` 的长序列
- 预处理选项: channel_norm（按被试减均值除标准差）、time_norm（按时间维度归一化）
- 默认: channel_norm=False, time_norm=False → **不做额外预处理**

### DualAlign Stage 1

```
数据来源: 认知负荷 EEG 数据 (EDF 文件)
格式:     加载后标准化为 (n_trials, n_channels, n_timepoints) float32
采样率:   256 Hz
epoch:    2 秒 (512 points)
```

- 加载后自动 preprocess: 带通滤波 + z-score 归一化
- 缓存到 .npz 文件避免重复处理

### 区别

| | CLISA | DualAlign |
|--|-------|-----------|
| EEG 段长度 | 5秒 (1250点) | 2秒 (512点) |
| 段来源 | 滑窗从30s视频切出 | 直接 epoch 切分 |
| 预处理 | 最小化（raw） | 带通滤波 + z-score |
| 数据标识 | 按 video 索引组织 | 按 (subject_id, label) 组织 |

---

## 2. 采样策略

### CLISA — `TrainSampler`

```python
# 1. 枚举所有被试对
for i in range(n_subs):
    for j in range(i+1, n_subs):
        sub_pairs.append([i, j])

# 2. 对每对被试
[sub1, sub2] = sub_pairs[s]

# 3. 从 28 个 video 中各取 1 个 segment
#    batch_size=28, n_videos=28 → n_samples_per_trial=1
for each video:
    ind_one = random.choice(该video的13个segment, 1)
ind_abs = 所有video采到的segment索引  # 共28个

# 4. 用相同索引分配给两个被试
ind_this1 = ind_abs + 364 * sub1
ind_this2 = ind_abs + 364 * sub2

# 5. 拼接
batch = [ind_this1(28个), ind_this2(28个)]  # 共56个
```

**关键特性**: 两个被试取的是**完全相同的 video 和 segment 位置**。

### DualAlign — `CrossSubjectPairDataset`

```python
# 1. 枚举所有被试对
pairs = list(combinations(unique_subs, 2))

# 2. 对每对被试
sub_a, sub_b = pairs[idx]

# 3. 按 label 均匀采样
per_label = samples_per_pair // n_labels  # 64 // 2 = 32

for each label:
    pool_a = index[(sub_a, label)]  # 被试A的该label的所有样本
    pool_b = index[(sub_b, label)]  # 被试B的该label的所有样本
    k = min(per_label, len(pool_a), len(pool_b))
    sel_a = random.choice(pool_a, k)  # 随机选k个
    sel_b = random.choice(pool_b, k)  # 随机选k个
```

**关键特性**: 两个被试取的是**同一 label 但不一定同一刺激时间段**的样本。

### 核心区别

| | CLISA | DualAlign |
|--|-------|-----------|
| 配对依据 | **相同刺激**(同video同时段) | **相同标签**(同condition) |
| 配对严格程度 | 严格：时间对齐 | 宽松：只需同label |
| batch 中 label 分布 | 不关心 label | 按 label 均匀采样 |
| 每个 batch 来自 | 恰好 2 个被试 | 恰好 2 个被试 |
| batch 大小 | 2 × 28 = 56 | 2 × 64 = 128 (可配置) |

---

## 3. 正负样本定义

### CLISA — InfoNCE

```
features: [z_A_0, z_A_1, ..., z_A_27, z_B_0, z_B_1, ..., z_B_27]  (56个)

labels = [0,1,...,27, 0,1,...,27]
label_matrix[i][j] = 1 iff labels[i] == labels[j]  (去掉对角线)
```

对于 anchor `z_A_i`（被试A的第i个video的segment）:
- **正样本**: `z_B_i` → 同一video同一segment，不同被试 → **1个**
- **负样本**: 所有其他 `z_A_j (j≠i)` 和 `z_B_j (j≠i)` → **54个**

**正样本的意义**: 同一刺激（同一视频同一时刻）在不同被试间应产生相似表征。

### DualAlign — SupConLoss

```
features = [z_A_0, ..., z_A_31, z_A_32, ..., z_A_63,
            z_B_0, ..., z_B_31, z_B_32, ..., z_B_63]  (128个)
           |-- label=0的32个 --|-- label=1的32个 --|  (被试A)
           |-- label=0的32个 --|-- label=1的32个 --|  (被试B)

all_labels = [labels_A, labels_B] = [0,...,0,1,...,1, 0,...,0,1,...,1]
pos_mask[i][j] = 1 iff all_labels[i] == all_labels[j] && i ≠ j
```

对于 anchor `z_A_k` (label=0):
- **正样本**: 所有其他 label=0 的样本，包括:
  - 被试A中其他 label=0 样本 (最多31个)
  - 被试B中所有 label=0 样本 (32个)
  → **共63个正样本**
- **负样本**: 所有 label=1 的样本
  - 被试A中 label=1 的 (32个)
  - 被试B中 label=1 的 (32个)
  → **共64个负样本**

**正样本的意义**: 同一认知负荷条件下（不管哪个被试、哪个时间段）的表征应该聚类在一起。

### 关键差异总结

| | CLISA | DualAlign Stage 1 |
|--|-------|-------------------|
| 正样本定义 | 时间对齐的跨被试对 | 同label的所有样本 |
| 正样本数量 | 每 anchor 恰好 1 个 | 每 anchor 多个 (N_same_label - 1) |
| 是否需要 label | **不需要** | **需要** |
| 学习目标 | 刺激-时间对齐 | 条件-类别聚类 |
| 负样本来源 | 不同 video 的所有 segment | 不同 label 的所有样本 |
| 对"正"的假设 | 相同外部刺激 → 相似神经响应 | 相同认知状态 → 相似神经响应 |

---

## 4. Loss 函数

### CLISA — InfoNCE (NT-Xent)

```python
# 相似度矩阵
similarity_matrix = features @ features.T          # (56, 56)

# 去掉对角线 → (56, 55)
# 正样本: 1列; 负样本: 54列
logits = [negatives(54), positives(1)] / temperature  # (56, 55)
labels = 54  (指向最后一列)

loss = CrossEntropyLoss(logits, labels)
```

- 本质是 (1 vs 54) 的分类问题
- 每个 anchor 独立计算，最终平均

### DualAlign — SupConLoss (Khosla 2020)

```python
sim = features @ features.T / temperature        # (128, 128)

# 去掉对角线
logits = sim - 1e9 * eye(128)
log_prob = logits - logsumexp(logits, dim=1)      # log-softmax

# 对每个anchor，对其所有正样本的 log_prob 取平均
mean_log_prob = (pos_mask * log_prob).sum(1) / n_pos

loss = -mean_log_prob.mean()
```

- 本质是对所有正样本 log-softmax 概率的平均值取负
- 当正样本 = 1 个时，退化为标准 NT-Xent (InfoNCE)

### 区别

| | CLISA InfoNCE | DualAlign SupConLoss |
|--|---------------|----------------------|
| 正样本处理 | 视为单一正类 → CE | 所有正样本 log-prob 取均值 |
| 当多正样本时 | 不适用（始终1个） | 自然支持 |
| temperature | 0.07 | 0.07 |
| 分母 | 所有非自身样本 | 所有非自身样本 |
| 准确率指标 | Top-1 / Top-5 | 最近邻检索准确率 |

---

## 5. 模型架构

### CLISA — ConvNet_baseNonlinearHead

```
Input: (B, 1, 30, 1250)

→ Stratified LayerNorm (per-subject, optional, at 'initial')
→ SpatialConv: Conv2d(1, 16, (30, 1))              # 空间滤波
→ Permute
→ TimeConv: Conv2d(1, 16, (1, 60), pad=29)          # 时间滤波
→ ELU
→ AvgPool: (1, 30)
→ Stratified LayerNorm (at 'middle1')
→ SpatialConv2: Conv2d(16, 32, (16, 1), groups=16)  # 分组卷积
→ TimeConv2: Conv2d(32, 64, (1, 6), groups=32)       # 分组卷积
→ ELU
→ Stratified LayerNorm (at 'middle2')
→ Flatten → 输出特征

无独立 projection head（特征直接用于对比学习）
```

**特色**: Stratified LayerNorm — 在 batch 内按被试分组做 LayerNorm，消除被试间幅度差异。

### DualAlign — EEGEncoder + ContrastiveProjector

```
Input: (B, 1, 32, 512)

→ ChannelAttention (SE-style, optional)
→ TemporalConv: Conv2d(1, 40, (1, 25))             # 时间滤波
→ BatchNorm2d + ELU
→ AvgPool: (1, 8), stride=(1, 4)
→ SpatialConv: Conv2d(40, 40, (32, 1))              # 空间滤波
→ BatchNorm2d + ELU + Dropout(0.5)
→ AdaptiveAvgPool2d((1, 16))
→ Flatten → Linear → embed_dim(256)

→ ContrastiveProjector:
  → Linear(256, 256) → ReLU → Linear(256, 128)     # 投影到对比空间
```

**特色**: Channel Attention + 标准 BatchNorm + 独立 projection head。

### 架构对比

| | CLISA | DualAlign |
|--|-------|-----------|
| 卷积顺序 | 空间→时间 | 时间→空间 |
| 注意力机制 | 无 | SE-style Channel Attention |
| 归一化 | Stratified LayerNorm (手工) | BatchNorm2d (标准) |
| 分组卷积 | 有 (depth-wise style) | 无 |
| Projection head | 无（直接用卷积输出） | 有 (2层MLP: 256→256→128) |
| Dropout | 无 (encoder中) | 有 (0.5) |
| 输出维度 | 卷积展平维度（非固定） | 128 (proj_dim) |

---

## 6. 训练循环

### CLISA

```python
for epoch in range(epochs_pretrain):        # 80
    for data, labels in train_loader:       # 每个batch=一对被试
        features = model(data)              # 56个特征
        logits, labels = info_nce_loss(features, stratified)
        loss = CE(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                    # 无梯度裁剪

    scheduler.step()                        # CosineAnnealingWarmRestarts
    # Early stopping: val accuracy 连续 30 epoch 不提升则停止
    # 保存 best model (按 val accuracy)
```

### DualAlign Stage 1

```python
for epoch in range(stage1_epochs):          # 100
    for batch in pair_loader:               # 每个batch=一对被试
        z_a = model.forward_cross_subject(eeg_a)
        z_b = model.forward_cross_subject(eeg_b)

        features = cat([z_a, z_b])          # 128个特征
        all_labels = cat([labels, labels])

        loss, acc = SupConLoss(features, all_labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

    scheduler.step()                        # CosineAnnealingLR
    # Early stopping: train loss 连续 15 epoch 不下降则停止
    # 保存 best model (按 train loss)
```

### 训练循环对比

| | CLISA | DualAlign Stage 1 |
|--|-------|-------------------|
| 最大 epoch | 80 | 100 |
| Optimizer | Adam (lr=7e-4, wd=0.015) | AdamW (lr=1e-3, wd=1e-2) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=epochs/3) | CosineAnnealingLR (T_max=epochs) |
| 梯度裁剪 | 无 | 有 (max_norm=1.0) |
| Early stopping 指标 | Val accuracy | Train loss |
| Patience | 30 | 15 |
| 验证集 | 有 (val_loader) | 无 (仅用 train loss) |
| Checkpoint | 保存 best + last | 保存 best + 每N个epoch |

---

## 7. 核心设计理念差异

### CLISA 的核心假设
> "相同外部刺激（同一视频同一时刻）在不同被试间应引发相似的神经响应。"

通过时间对齐的跨被试对比，学习**跨被试不变的刺激驱动表征**。不需要任何标签信息——刺激的时间对齐本身就提供了自监督信号。

### DualAlign Stage 1 的核心假设
> "相同认知负荷条件下（无论被试、时间段），EEG表征应聚集在一起。"

直接利用标签信息进行有监督对比学习，学习**跨被试的条件判别性表征**。label 信息使得正样本定义更宽泛（任意同条件样本），但也引入了对标注的依赖。

### 总结一句话

- **CLISA**: 用「刺激时间对齐」做自监督 → 每个anchor 1个正样本 → InfoNCE
- **DualAlign Stage 1**: 用「条件标签」做有监督 → 每个anchor 多个正样本 → SupConLoss
