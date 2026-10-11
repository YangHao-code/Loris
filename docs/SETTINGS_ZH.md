# 参数

| 阶段 | 参数 |
|---|---|
| 选择器 | `global_margin_macro_f1`＋`pooled_margin_macro_f1`；K=4；100 epochs；前25轮仅模仿；hidden=128；MC=64；σ=0.1；lr=0.001；dropout=0.1；seed=42 |
| 选择器损失 | 模仿权重2；任务权重0.1；熵权重0.01；任务为0.8 soft-macro＋0.2 soft-micro |
| 单专家阈值 | 网格0.025…0.975，步长0.025；收缩权重 `n_pos/(n_pos+20)`；零正例保留全局阈值 |
| 组合谓词 | `sigmoid(mean(logit(p_m)-logit(cut_m,label)))`；规则阈值0.5；裁剪ε=1e-6 |
| AAPD发现 | `mode=diverse`；seed=28101；A内三份，2/3生长、1/3接纳；随后B剪枝 |
| BGC发现 | `mode=frozen`；seed=25101；比较文本→传播、传播→文本及单阶段 |
| 原BO | JointBinaryTPE；startup=12；expected body size=1.5；ML cut∈[0.001,0.999] |
| diverse搜索 | 每类型平均16 trials/label＋预热；每标签至少8；额外预算60%残余误差、40%√训练支持量 |
| frozen搜索 | 每阶段2×模板数＋4×标签数；沿用原文本/关系守卫 |
| diverse目标 | 0.9 macro＋0.1 micro；A内macro不降，micro相对初始最多降0.002 |
| frozen目标 | 0.8 macro＋0.2 micro；两指标不降；三块稳定性项；不是独立交叉验证 |
| 最终选择 | `priority=diversity`；按移除整类后的必要TP计有效类型；默认B macro容忍0、micro容忍0.002，可显式调整 |
| 替换 | `allow_replacement=true`；仅发现阶段；更新后重求闭包撤回失去支持的事实 |
| Aux | 发布沿用独立正负规则与校准门；本次指标不含新Aux训练；不宣称已有逐轮稳定提升 |

## 输入

`staged-rules` 接受可信本机生成的 context pickle、POOL.json、旧规则JSON、原谓词库存和分区NPZ；文件均由SHA绑定。

分区NPZ：`row_ids`、`labels`、`A`、`truth_A`、`B`、`truth_B`。A/B为context行下标，真值为完整bool矩阵；只允许dev/calibration行，不能进入Gamma。发现context只含train/dev。`diverse`的B参与拟合，不是独立验证集。

新数据先用 `loris.discovery.staged.pool.fit_pool(...)` 训练选择器；输出POOL.json和完整选择器记录。复用已学规则时必须使用同一模型检查点、标签轴和数据版本，不能仅凭模型名复现分数。

## 已发布配置

| 数据集 | 实际选出的四个冻结专家 | 默认规则／参照 |
|---|---|---|
| AAPD | DeBERTa-v3-large ASL、Encoder-MLP、RoBERTa-large wBCE、word+char LR | `aapd_diverse_rules.json`／`aapd_quality_rules.json` |
| BGC | DeBERTa-v3-large ASL、MiniLM微调、RoBERTa-large ASL、RoBERTa-large wBCE | `bgc_diverse_rules.json`／`bgc_quality_rules.json` |

文件在 `configs/verified/`；这里只存规则和数值参数，不存权重、数据或服务器配置。
