# 已覆盖文档的查询与复核

一个正标签不代表整份多标签答案完整。`loop.query_pool` 支持：

- `uncovered`：保留原来只查询未覆盖文档的行为。
- `all_unanswered`：查询角色中尚无完整可信答案的文档，即使已有逻辑正标签。
- `needs_review`：使用 `query.strategy: "review"` 排序，并按声明的比例随机审计其余未完整回答的文档。

查询角色由输入上下文显式指定，可以是目标标注集；它不由变量名 `training` 决定。查询后用于训练的行可以包含授权获取答案的目标文档，但未询问的真值不会进入选样或训练。

## 按证据复核

以下片段应合并到已有完整 manifest，其他预算和反馈设置仍需明确填写：

```json
{
  "loop": {
    "query_pool": "needs_review",
    "query_cadence": "each_closure",
    "query_batch_fraction": 0.01,
    "query_batch_min": 32,
    "query_answer_mode": "complete"
  },
  "query": {
    "strategy": "review",
    "seed": 0,
    "review": {
      "eligibility_mode": "evidence",
      "near_boundary_ratio": 0.9,
      "audit_fraction": 0.125
    }
  }
}
```

`evidence` 模式优先复核未完整回答且符合以下条件的文档：没有正标签、仍有未仲裁的逻辑冲突、模型与当前输出在未知真值格上分歧，或标签数低于训练平均数且至少一个缺失标签的模型分数接近其部署阈值。

最后一种情况要求 `0.9 × κ ≤ p < κ`，且 `κ > 0`。标签少本身不是缺标签的证据，因此原本只有一两个标签、其他标签分数远低于阈值的文档不会仅因标签数少而被选中。已获可信答案的格不参与分歧和近阈值判断；已有部分答案的文档仍可复核，完整可信答案的文档排除。

训练平均标签数只从指定训练角色中有完整可信答案的行计算，不包含目标集已获取的答案。部署阈值是谓词判定阈值，不是经过独立认证的正确率。

`audit_fraction` 默认 `0.125`。随机审计涵盖没有风险证据的未完整回答文档；风险候选不足时，也可用审计补足批量。设为 `0` 则只使用证据候选。

省略 `eligibility_mode` 时保留 `legacy` 行为：标签数少于 `cardinality_ratio × 训练平均数` 也可成为候选，默认比例 `0.7`。这不是要求“70% 的文档标签不足”，而是每篇文档的标签数门槛；`evidence` 模式不使用该比例作为独立查询条件。

## 批大小和预算

未声明固定 `query_batch_size` 时：

```python
requested = max(query_batch_min, ceil(initial_pool_size * query_batch_fraction))
count = min(requested, current_eligible_documents, remaining_total_budget)
```

实际送去询问的是 `count` 篇。`initial_pool_size` 始终是启动时查询角色中尚无完整可信 Γ 答案的文档数，不会因 `uncovered` 或风险证据筛选而缩小；当前候选数只限制 `count`。无总预算上限时省略最后一项。默认每批至少 32、初始未完整回答文档数的 1%；最后一批可以少于 32。

预算按请求文档计数，同一请求从 LLM 转人工只占一个请求预算。调用来源、LLM 接受和人工回退分别记录；模拟 GT 答案必须显式声明，不能报告为真人标注。完整答案、部分答案和仍未知的格分别保留，未知格不作为负例。

这些选项扩展查询资格，没有新增基于 test F1 的停止条件。预算耗尽、候选耗尽及原有运行终止协议继续生效。若目标标注集包含 test 文档，应分别报告全体结果和未直接询问文档的结果。
