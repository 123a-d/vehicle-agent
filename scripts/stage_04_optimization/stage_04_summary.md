# Stage 4：模型优化、Unknown 专项实验与最终模型选择

## 1. 阶段目标

本阶段以 Stage 3 Baseline 为起点，继续进行 Fine-tuning、Unknown 专项实验和最终模型选择。

Baseline：

```text
Macro-F1 = 0.6017
Unknown F1 = 0.0241
```

主要目标：
1. 提升 50 个目标车型的识别能力。
2. 改善 Unknown 类别表现。
3. 分析开放集拒识的实际能力边界。
4. 选择后续推理层和 Agent 使用的最终模型。

## 2. 实验总览

| 实验 | 方法 | Val Acc | Macro-F1 | Known Macro-F1 | Unknown F1 | 结论 |
|---|---|---:|---:|---:|---:|---|
| Baseline | 冻结 Backbone，仅训练 fc | 0.5995 | 0.6017 | - | 0.0241 | 基线 |
| Exp 1 | 解冻 layer4 + fc | 0.8139 | 0.8239 | - | 0.1458 | 大幅提升 |
| Exp 2 | 数据增强 | - | 0.8137 | - | 0.0706 | 不采用 |
| Exp 3 | layer3 + layer4 + fc，分层 LR | - | 0.8254 | - | 0.1443 | 提升很小 |
| Exp 4A | Unknown Weighted CE | 0.8308 | **0.8417** | **0.8544** | **0.2056** | 最终采用 |
| Exp 4C | Unknown 多样性扩充 | - | 0.8191 | - | 0.1124 | 下降 |
| Exp 4E | Hard Negative v3 | - | 0.8212 | - | 0.1538 | 未超过 4A |

## 3. Exp 1：解冻 layer4 + fc

训练策略：从 Baseline checkpoint 出发，解冻 `layer4 + fc`，其余 backbone 保持冻结。

主要设置：
- Learning Rate：`1e-4`
- Epochs：5
- 不使用额外数据增强
- layer4 中 BatchNorm 保持 eval 状态

最佳结果：

| 指标 | 结果 |
|---|---:|
| Validation Accuracy | 0.8139 |
| Macro-F1 | 0.8239 |
| Unknown F1 | 0.1458 |

相比 Baseline：

```text
Macro-F1
0.6017 → 0.8239
```

## 4. Exp 2：数据增强

数据增强包括：
- RandomResizedCrop，scale = 0.8 ~ 1.0
- RandomHorizontalFlip，p = 0.5
- ColorJitter

结果：

```text
Macro-F1 = 0.8137
Unknown F1 = 0.0706
```

相比 Exp 1 下降，因此未采用。

## 5. Exp 3：解冻 layer3 + layer4 + fc

分层学习率：

```text
layer3 : 1e-5
layer4 : 5e-5
fc     : 1e-4
```

最佳结果：

```text
Macro-F1 = 0.8254
Unknown F1 = 0.1443
```

相比 Exp 1 仅提升约 `0.0015`，收益很小。

## 6. Exp 4A：Unknown Weighted Cross Entropy

### 6.1 方法

由于 Unknown 类别明显较弱，提高 Unknown 类别在 CrossEntropyLoss 中的权重。

```python
class_weights = torch.ones(51)
class_weights[50] = 2.0
```

即 Unknown 误分类时的损失权重为普通类别的 2 倍。

### 6.2 最佳结果

| 指标 | 结果 |
|---|---:|
| Validation Accuracy | 0.8308 |
| Macro-F1 | **0.8417** |
| Known Macro-F1 | **0.8544** |
| Unknown Precision | 0.2340 |
| Unknown Recall | 0.1833 |
| Unknown F1 | **0.2056** |

最终 checkpoint：

```text
D:\car_project\outputs\unknown_weighted\best_macro_f1.pth
```

## 7. Exp 4B：Confidence Threshold 分析

### Threshold = 0.40

| 指标 | 结果 |
|---|---:|
| Macro-F1 | 0.8476 |
| Known Macro-F1 | 0.8600 |
| Unknown Precision | 0.1932 |
| Unknown Recall | 0.2833 |
| Unknown F1 | 0.2297 |
| Known Rejection Rate | 0.0377 |

### Threshold = 0.55

| 指标 | 结果 |
|---|---:|
| Macro-F1 | 0.8477 |
| Unknown Recall | 0.4333 |
| Unknown F1 | 0.2047 |
| Known Rejection Rate | 0.0891 |

很多真正的 Unknown 即使被错误分类为已知车型，也具有较高置信度：

```text
Mean   ≈ 0.7027
Median ≈ 0.7380
```

因此单纯依赖 Confidence Threshold 只能做有限的业务层风险控制。

## 8. Exp 4C：Unknown 多样性扩充

构造：

```text
100 个非目标 source classes × 每类 2 张 = 200 Unknown samples
```

结果：

```text
Macro-F1 = 0.8191
Unknown F1 = 0.1124
```

未超过 Exp 4A。

## 9. Exp 4D：Hard Negative Mining

扫描非目标车型，寻找容易被模型高置信度误识别为已知类的 Unknown source。

统计：

```text
Mean Known Confidence >= 0.70：304 个 source
Mean Known Confidence >= 0.90：55 个 source
```

说明 Hard Unknown 大量存在。

## 10. Exp 4E：Hard Unknown v3

构造：

```text
40 个 Hard source × 4
+
20 个 Regular source × 2
=
200 Unknown samples
```

结果：

```text
Macro-F1 = 0.8212
Unknown F1 = 0.1538
```

仍未超过 Exp 4A。

## 11. Exp 4F：Prototype Cosine Rejection

Known 正确预测样本：

```text
Mean   = 0.9361
Median = 0.9398
Q25    = 0.9245
```

True Unknown 但误分类为 Known：

```text
Mean   = 0.9296
Median = 0.9356
Q75    = 0.9457
```

两个分布高度重叠，最佳阈值接近 `0.00`，因此 prototype cosine similarity 无法有效分离 Known 与 Unknown。

## 12. Exp 4G：独立 Known / Unknown Gate

Gate：

```text
512-d feature
  ↓
Linear(512, 1)
  ↓
Known / Unknown
```

### threshold = 0.50

| 指标 | 结果 |
|---|---:|
| Macro-F1 | 0.5350 |
| Known Macro-F1 | 0.5443 |
| Unknown Precision | 0.0378 |
| Unknown Recall | 0.6500 |
| Unknown F1 | 0.0715 |
| Known Rejection Rate | 0.5263 |

在限制 Known Rejection 约 5%～10% 时：

```text
threshold = 0.05
Macro-F1 = 0.8348
Known Macro-F1 = 0.8515
Unknown F1 = 0.0
```

因此独立 Gate 也未解决问题。

## 13. 最终模型选择

最终采用：

```text
Exp 4A：Unknown Weighted Cross Entropy
```

最终模型：

```text
D:\car_project\outputs\unknown_weighted\best_macro_f1.pth
```

核心指标：

```text
Validation Accuracy = 0.8308
Macro-F1            = 0.8417
Known Macro-F1      = 0.8544
Unknown F1          = 0.2056
```

整体优化路线：

```text
Baseline
Macro-F1 0.6017
      ↓
layer4 Fine-tuning
Macro-F1 0.8239
      ↓
Unknown Weighted CE
Macro-F1 0.8417
```

## 14. Unknown 能力边界

本阶段实验说明：当前模型主要适合 50 个目标车型识别，对真正 out-of-scope 的未见 Unknown 车型拒识能力仍有限。

Confidence、Prototype Similarity、独立 Gate 都无法稳定解决开放集识别问题。

因此后续系统不会把模型描述为通用开放集车辆识别器。

## 15. Agent 层工程处理

后续推理层与 Agent 保留：

```text
confidence
status
warning
```

通过 `recognized / uncertain / unknown` 等状态避免对低置信度或 Unknown 风险样本做过度承诺。

## 16. Stage 4 最终结论

```text
✅ 停止继续进行 Unknown 算法研究
✅ 保留 Unknown Weighted CE 模型
✅ best_macro_f1.pth 作为后续最终基础模型
✅ 明确记录开放集 Unknown 能力边界
✅ 后续重点转向推理模块、Tool 和 Agent 应用开发
```

Stage 5 接口：

```text
Checkpoint
  ↓
Inference Module
  ↓
Visual Tool
  ↓
Agent
```
