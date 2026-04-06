# Stage 1 向 CLISA 靠拢 — 已实施的修改记录

## 修改总览

| 修改项 | 文件 | 变更内容 |
|--------|------|----------|
| Config 新增参数 | `config.py` | 新增 `epoch_step=0`(=epoch_sec), `stage1_segs_per_cond=14`; epoch_sec 保持 2.0 |
| 预处理 z-score | `base_loader.py` | **保留不变** — z-score 与 StratifiedLayerNorm 互补（不同维度） |
| 滑窗切段 + 位置索引 | `loaders/eegmat.py` | `_segment_epochs` 支持 overlap；保存 epoch `positions`；默认无重叠 |
| Stratified LayerNorm | `models/encoder.py` | 新增 `StratifiedLayerNorm`，插入 temporal→spatial 之间 |
| Encoder forward 参数 | `models/encoder.py`, `models/dual_align.py` | 接受 `n_per_subject`，Stage 1 激活，其他阶段为 None |
| CLISA 式采样器 | `data/dataset.py` | `CrossSubjectPairDataset` 按位置对齐采样，两被试共享 epoch 索引 |
| InfoNCE Loss | `trainers/stage1.py` | SupConLoss → InfoNCELoss，不再传入 labels |

---

## 详细变更

### 1. 切段参数

```
默认: epoch_sec=2.0, epoch_step=0 (=epoch_sec, 无重叠) → 与原来完全一致
可调: epoch_sec=4.0, epoch_step=1.0 → 更长窗口 + 重叠 (需通过CLI参数设置)
```

- 默认保持不变，不影响 Stage 2/3 的数据
- 数据加载新增 `positions` 字段记录 epoch 在条件内的时间位置
- 如需更长窗口/重叠滑窗，通过 `--epoch_sec 4.0 --epoch_step 1.0` 启用
- 注意：改变 epoch 参数会影响所有阶段的输入数据

### 2. 归一化链路 (互补设计)

```
所有阶段: 带通滤波 → z-score (保留，沿时间轴归一化)
Stage 1:  + StratifiedLayerNorm (模型内，沿特征维度按被试分组归一化)
Stage 2/3: StratifiedLayerNorm = identity (不激活)
```

- z-score 与 StratifiedLayerNorm 作用于不同维度，不冗余
- z-score: 每 epoch 每通道，消除直流偏移和量级
- StratifiedLayerNorm: batch 内按被试分组，消除跨被试的系统性差异
- Stage 2/3 的归一化链路与改动前完全一致

### 3. 采样策略 (CLISA 式)

```
原来: 枚举被试对 → 按 label 均匀采样 → 两被试独立选择样本
现在: 枚举被试对 → 从每个条件选 14 个共同位置 → 两被试用相同的 epoch 位置
```

Batch 结构 (segs_per_cond=14, 2 条件):
```
eeg_a = [rest_seg_p0, ..., rest_seg_p13, arith_seg_q0, ..., arith_seg_q13]  ← 28 个
eeg_b = [rest_seg_p0, ..., rest_seg_p13, arith_seg_q0, ..., arith_seg_q13]  ← 28 个 (相同位置!)
```

InfoNCE 中:
```
z_a = encoder(eeg_a)  # (28, proj_dim)
z_b = encoder(eeg_b)  # (28, proj_dim)
features = cat([z_a, z_b])  # (56, proj_dim)

正样本: z_a[i] ↔ z_b[i]  (同条件同位置不同被试) → 1 个/anchor
负样本: 所有其他 z_a[j], z_b[j] (j≠i) → 54 个/anchor
```

### 4. Loss 变更

```
原来: SupConLoss — 所有同 label 样本为正样本（多正样本）
现在: InfoNCELoss — 只有位置对齐的跨被试对为正样本（单正样本）
```

---

## 已知局限与设计取舍

### 正样本假设较弱

EEGMAT 缺乏共享刺激，"同一位置" ≠ "相同神经活动"。
但 60s 连续任务中存在部分共性时间演化（启动/稳态/疲劳），
位置对齐提供了一个弱但非零的辅助信号。

### 同条件内 segment 被当作负样本

rest_seg_3 和 rest_seg_7 都是休息态但被视为负样本。
由于同条件 segment 高度相似，它们的相似度高、loss 贡献小，
模型主要从 rest vs arith 的差异中学习，影响有限。

### 数据量仍然有限

36 个被试 → ~630 个被试对（C(36,2)），远少于 CLISA 的 5995 对。
通过增大 segs_per_cond 和重叠滑窗部分缓解。
