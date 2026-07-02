# **LORIS 框架综合实验评估清单**

本文档详细梳理了《Logic \+ ML for Human-Efficient Document Labeling》第七部分中提及的所有实验细节 \[cite: 509, 510\]。为了全面评估系统在端到端标注、人工成本、系统效率及模型选择上的表现，需要执行以下五个核心实验及相关基线对比。

## **一、 实验基础配置 (Setup & Datasets)**

实验需要部署在统一的数据集和模型池上，涵盖临床、金融、新闻和学术等多种不同领域和规模的数据。

### **1\. 评估数据集**

| 数据集名称 | 所属领域 |
| :---- | :---- |
| MIMIC-III-50 \[cite: 512\] | Clinical (临床医疗) \[cite: 512\] |
| Reuters-21578 \[cite: 512\] | Finance (金融) \[cite: 512\] |
| RCV1-V2 \[cite: 513\] | News (新闻) \[cite: 513\] |
| arXiv-Full \[cite: 514\] | Academic (学术) \[cite: 514\] |

### **2\. 模型池 (Model Pool)**

模型池需要包含四个层级的预训练和微调模型，覆盖不同的计算成本和语义复杂度：

* 传统特征模型：TF-IDF 结合 Logistic Regression 或 SVMs \[cite: 515\]。  
* 深度神经网络 (DNNs)：TextCNNs，以及 RNNs/BiLSTMs \[cite: 516\]。  
* 预训练编码器：RoBERTa, DeBERTa 结合 MLP 或 XGBoost 分类器 \[cite: 517\]。  
* 微调小语言模型 (SLMs)：使用 LoRA 或 adapter 微调的 Llama-3-8B 和 Mistral-7B \[cite: 518\]。

### **3\. 默认参数与标注器配置**

* 超参数控制：需要调整和记录的模型数量 K、模型池大小 |M|、噪声采样 N、平滑温度 ε，以及损失权重 λ\_task 和 λ\_ent \[cite: 539\]。  
* LLM 标注器：指定为 gpt-4.1-2025-04-14 (temperature=0) \[cite: 540\]。  
* 数据划分：明确执行 Train/Val/Test 数据集的划分 \[cite: 541\]。

### **4\. 备选评估数据集分析**

根据多标签文本分类（MLTC）的标准以及实际文本获取的难度，对以下五个备选数据集进行了详细的核查与分析，以供后续实验扩展参考：

| 数据集 | 是否官方 MLTC | 原始文本是否直接提供 | 获取难度 | 详细评价 |
| :---- | :---- | :---- | :---- | :---- |
| arXiv Metadata | 是 | 是 | ⭐ 极易 | 标签天然（如 cs.CL, cs.LG 等），标题和摘要直接可用，Kaggle 即可获取 JSON 文件，是最理想的 MLTC 基准。 |
| HUPD | 是 | 是 | ⭐ 极易 | 多标签来自 IPC 或 CPC，官方直接提供完整的专利文本（标题、摘要、权利要求等），文本较长，极其适合评估大语言模型。 |
| PubMed MeSH | 是 | 是 | ⭐⭐ 较易 | 经典的 XMLC 数据集，论文天然具备多个 MeSH 主题词标签。原始摘要容易获得，但不同来源（如 BioASQ, PMC 等）的数据需要少许预处理和对齐工作。 |
| Goodreads Book Graph | 可构造 | 部分提供 | ⭐⭐⭐ 中等 | 本身为涵盖推荐、图学习等多种任务的数据集集合。可通过 popular shelves 构造 MLTC 任务，但官方并未将其直接作为 MLTC 基准。且因版权原因，部分文本（如书籍简介）可能缺失。 |
| MICoL MAG-CS | 可构造 | 否（需映射） | ⭐⭐⭐⭐ 偏难 | 主要用于多模态图表示学习，可通过 Field of Study 构造多标签。但由于 MAG 已经退役，需将 ID 映射至 OpenAlex，且许多摘要以倒排索引（Inverted Index）形式存储，需额外编写脚本恢复文本，数据对齐工作繁琐。 |

**实验优先级建议：**

* **arXiv：** 标签天然、多标签明确、标题和摘要直接可用，文本获取最省心。  
* **HUPD：** 专利文本完整且较长，适合研究长文本多标签分类。  
* **PubMed MeSH：** 经典 XMLC 基准，医学领域权威，但标签体系和预处理稍复杂。  
* **Goodreads：** 更适合作为可自行构建 MLTC 任务的数据源，而不是直接使用的官方 MLTC 基准。  
* **MICoL MAG-CS：** 更偏向图学习与学术网络分析，如果你的研究重点是文本分类，它并不是最省力的选择，因为文本恢复和数据对齐工作会明显更多。

## **二、 核心实验矩阵 (Core Experiments)**

下面是各项对比与消融实验的具体方案：

### **Exp-1: LORIS Accuracy (端到端标注准确性)**

* 核心指标：端到端 Macro-F1 分数 \[cite: 533, 545\]。  
* 对比基线：Snuba, Self-Pretraining, RulePrompt, DeBERTa\_SVM, DeBERTa\_XGBoost, GPT4, BESRA, 以及 RAL \[cite: 545, 546\]。  
* 影响估计 (Influence Estimation) 评估：使用 MRR (Mean Reciprocal Rank) 指标，对比 RDG 估计机制的效果 \[cite: 549, 567\]。

### **Exp-2: LORIS Human Cost (人工标注成本降低效果)**

* 核心指标：完成全量文档标注所需的人工干预次数 (\# Human annotations) \[cite: 572\]。  
* 对比基线：需要人工在环 (HITL) 交互的基线模型 BESRA 和 RAL \[cite: 571, 572\]。  
* 控制变量 1：改变文档集大小 |D| (20% 到 100%)，测试规模扩展时人工成本的增长曲线 \[cite: 575\]。  
* 控制变量 2：改变规则集大小 |Σ| (20% 到 100%)，评估发现的规则数量对进一步降低人力成本的作用 \[cite: 579, 580\]。  
* 控制变量 3：改变初始真实标签数量 |Γ| (Varying |Γ|)，观察提供给追逐算法的初始先验数据量的影响 \[cite: 583\]。

### **Exp-3: LORIS Scalability (系统计算效率与可扩展性)**

* 核心指标：端到端标注运行时间 (Time) \[cite: 587\]。  
* 控制变量 1：文档规模 |D| 变化 (20% 到 100%) 条件下的运行时间开销扩展性 \[cite: 590, 591\]。  
* 控制变量 2：规则集规模 |Σ| 变化 (20% 到 100%) 条件下的处理时间开销 \[cite: 593, 594\]。

### **Exp-4: Model Selection (动态路由与模型选择模块)**

* 核心指标：所选模型组合在下游任务的 F1 分数，以及选择模块自身消耗的时间 \[cite: 534, 596, 618\]。  
* 对比基线：RandomMs, IndivMs, Hybrid\_LLM, 以及 CAAS \[cite: 520, 521, 522, 526\]。  
* 控制变量 1：改变每次路由选择的模型数量 K (设置从 1 逐渐增加到 |M|) \[cite: 599\]。  
* 控制变量 2：改变整体候选模型池的大小 |M| (Varying |M|) \[cite: 613\]。

### **Exp-5: Ablation Study (核心架构消融实验)**

* 禁用 LLM (LORIS\_noLLM)：关闭 LLM 辅助标注器，在不确定的情况下完全依赖人工，以验证大模型作为初筛机制带来的降本效果 \[cite: 625, 626\]。  
* 禁用增量追逐 (LORIS\_noInc)：将增量追逐 (incremental chasing) 替换为一次性全局生成所有估值，证明增量策略在系统效率上的绝对必要性 \[cite: 627, 628\]。  
* 替换梯度近似 (LORIS\_noS)：移除现有的随机平滑 (stochastic smoothing) 方法，替换为标准的 Gumbel-Softmax 弛豫，观察训练稳定性和准确度变化 \[cite: 629\]。  
* 调整损失函数 (LORIS\_noL)：仅使用下游任务损失 (task loss) 进行训练，移除联合模仿损失，证明混合优化的必要性 \[cite: 631\]。  
* 模式选择策略替换：将当前的基于判别力的模式选择方法替换为 Filter\_MI, Filter\_Chi2, WeShap 和 LocalBoost，对比 F1 分数的下降幅度 \[cite: 632, 633, 634\]。