# LORIS

多标签文档标注：**规则产生初始闭包 → 独立 Aux 动态规则 → 校准后迭代**。当前发布优先规则多样性，保留高分配置。

## 谓词池

| 类别 | 谓词／规则示例 |
|---|---|
| 模型 | `M(x,A) ≥ τ → A(x)`；可用单专家或校准后的专家组合 |
| 多模型 | `M₁(x,A) ≥ τ₁ ∧ M₂(x,A) ≥ τ₂ → A(x)` |
| 文本 | 词频、匹配、共现、距离、依存结构等；复用原谓词库存 |
| 模型＋文本 | `Text(x) ∧ M(x,A) ≥ τ → A(x)` |
| 标签传播 | `B(x) ∧ 条件 → A(x)` |
| 跨文档传播 | `关联(x,y) ∧ B(y) ∧ 条件 → A(x)` |
| 动态 Aux | 独立模型的高阈值正规则、低阈值负规则；旧专家及静态规则冻结 |

模型候选池共 **16 个**：DeBERTa-v3 base/large、RoBERTa-large ASL/wBCE、MiniLM 微调/冻结线性/冻结 MLP、Encoder-MLP、BiLSTM、TextCNN、Aux 初始快照，以及 TF-IDF LR、TF-IDF SVM unigram/bigram、NB-SVM、word+char LR。原 **MLP＋MC Top-K** 选择器学习组合，原 **JointBinaryTPE BO** 搜索规则。

## 已核验表现

完整 test，LLM／真人／GT模拟人工均为 0；本表未叠加新 Aux 训练。发布配置取自已完成的冻结候选；历史 dev 已复用，AAPD 多样性版的 B 参与剪枝。

| 数据集／配置 | 规则组成 | macro-F1 | micro-F1 | FP | FN | Comp |
|---|---|---:|---:|---:|---:|---:|
| **AAPD 多样性优先** | 55 模型＋2 多模型＋2 文本＋1 模型文本 | **0.62125** | 0.76321 | 517 | 608 | 0.77763 |
| AAPD 高分参照 | 54 模型 | 0.62959 | 0.77128 | 480 | 600 | 0.78055 |
| **BGC 多样性优先** | 146 模型＋42 模型文本＋8 标签传播＋29 跨文档 | **0.70829** | 0.83323 | 9738 | 8862 | 0.85085 |
| BGC 高分参照 | 146 模型＋8 标签传播＋29 跨文档 | 0.70934 | 0.83419 | 9295 | 9087 | 0.84722 |

“模型”按一个 ML 原子计，可能引用四专家组合。Comp＝真值标签数大于 1 的文档上，逐文档标签召回率的均值。

[完整指标](docs/results/verified_rules_20261011.csv) · [规则与阈值](configs/verified/) · [主要参数](docs/SETTINGS_ZH.md)

## Setting

| 项目 | 设置 |
|---|---|
| 数据 | AAPD：train 53840／test 1000／54 labels；BGC：58715／18394／146 labels |
| 专家训练 | 完整 train；arXiv 沿用固定 10% train；不改变 dev/test |
| 选择器 | K=4，hidden=128，epochs=100，MC=64，σ=0.1，lr=0.001，seed=42 |
| 专家组合 | 每标签校准阈值；均值 logit margin；稀有标签阈值向全局收缩 |
| 规则搜索 | 原 BO；分类型搜索；完整闭包打分；无每标签最终条数上限 |
| 接纳 | 有效类型优先，dev macro/micro 约束；替换可选；B剪枝须标明拟合用途 |
| Aux | 独立别名＋逐标签高低阈值；只更新新增动态部分；完整闭包校准门 |
| 半监督 | 目标文本＋可靠软伪标签＋train 回放；未知格不计损失 |
| 查询 | 已有正标签仍可复核；按分歧/冲突等选样，保留预算约束 |
| 加速 | 复用专家概率和谓词；稀疏矩阵筛选＋位集合闭包；原生 chase 核验 |

## 使用

```bash
python -m pip install -e ".[dev,aligned]"
export LORIS_SSL_SOURCE="$PWD/backends/transformer/src"
python -m pytest tests
loris-discover staged-rules --manifest staged.json --output runs/staged
```

用 [输入模板](configs/templates/staged_discovery.json) 填写本机资产路径与 SHA；数据、模型权重由调用方提供。已学规则可配合 `loris.discovery.staged.pool.extend_context` 生成组合分数，再交给原 chase。

[动态 Aux](docs/residual_aux.md) · [查询策略](docs/QUERY_POOL_ZH.md) · [既有接口](docs/POST_V1_UPDATES.md)。校准通过不保证 test 每轮单调上升。
