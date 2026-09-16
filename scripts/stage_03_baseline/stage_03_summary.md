# Stage 3：Baseline ResNet18 训练与评估总结

## 1. 阶段目标

建立一个可复现的 51 类车辆识别 Baseline，作为后续模型优化的统一对照基准。

分类任务包含：
- 50 个目标车型类别
- 1 个 `unknown` 类别
- `unknown` 类别索引为 `50`

## 2. 模型设计

使用 ImageNet 预训练的 `ResNet18`。

训练策略：
- 冻结 backbone
- 将最后的全连接层替换为 51 类输出
- 只训练新的 `fc`

结构：

```text
输入图片
  ↓
ResNet18 Backbone
  ↓
51 类 fc
  ↓
51 类 logits
```

## 3. 训练配置

| 参数 | 设置 |
|---|---|
| Random Seed | 42 |
| 输入尺寸 | 224 × 224 |
| Batch Size | 32 |
| Epochs | 5 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Train Samples | 5902 |
| Val Samples | 1945 |
| Classes | 51 |
| Unknown Index | 50 |

## 4. Baseline 结果

最佳结果出现在 Epoch 5。

| 指标 | 结果 |
|---|---:|
| Validation Accuracy | 0.5995 |
| Macro-F1 | 0.6017 |
| Unknown Precision | 0.0435 |
| Unknown Recall | 0.0167 |
| Unknown F1 | 0.0241 |

Baseline 已具备基本识别能力，但 Unknown 类别识别明显较弱。

## 5. 模型文件

Checkpoint：

```text
D:\car_project\outputs\baseline_resnet18\baseline_best.pth
```

类别映射：

```text
D:\car_project\outputs\baseline_resnet18\class_mapping.json
```

## 6. 核心结论

Baseline 的 `Macro-F1 = 0.6017`，可作为后续实验对照基准。

主要问题：
1. 冻结 backbone 后，对当前车辆数据集适应能力有限。
2. Unknown F1 仅为 `0.0241`。
3. 后续需要解冻高层网络进行 Fine-tuning。
4. 后续需要针对 Unknown 做专项优化。

## 7. Stage 4 接口

```text
Baseline ResNet18
  ↓
解冻高层网络
  ↓
Fine-tuning
  ↓
Unknown 专项优化
  ↓
最终模型选择
```

本阶段状态：

```text
✅ Baseline 训练完成
✅ checkpoint 保存完成
✅ 重新加载与评估完成
✅ 为 Stage 4 提供统一基线
```
