# simple_data/：五个领域的最终数据集，以及各 1,500 训练 + 500 评测的抽样

每个领域只用该领域唯一的训练集，去掉与该领域所有评测集的重合部分和训练集内部的重复题，得到最终数据集
（`pool.jsonl`），再从中随机抽 1,500 道作训练、500 道作评测。
生成脚本：`data/scripts/make_simple_data.py`（种子 `20261118:simple_data:<domain>`，可重复生成）。

```bash
eval/.venv/bin/python -I data/scripts/make_simple_data.py                  # 全部领域
eval/.venv/bin/python -I data/scripts/make_simple_data.py --only math,code # 部分领域
```

## 文件

```
simple_data/<domain>/pool.jsonl         最终数据集（去污染、去重后的全部题，按 id 排序；共约 210 MB，不进 git，用脚本重新生成）
simple_data/<domain>/train.jsonl        1,500 条，从 pool.jsonl 抽
simple_data/<domain>/eval.jsonl           500 条，从 pool.jsonl 抽，与 train 不重叠
simple_data/<domain>/build_report.json  每一步的数量、按评测集统计的重合、被移除的 id
```

`pool.jsonl` 与 `data/processed/<训练集>.jsonl` 中 `split_pool == "pool"` 的部分完全相同，也就是正式运行用的抽样池。
schema 与 `data/processed/*.jsonl` 相同（`id, dataset, category, split_pool, prompt, messages, reference, metadata`），
train / eval 文件的 `split_pool` 改写为 `train` / `eval`。代码题的测试用例仍在 `data/processed/aux/taco_verified_tests.jsonl`，按 `id` 关联。

## 来源与对照的评测集

复查对照的是同类别每个评测集的**全部**样本（test 和 reserve 都算）。

| 领域 | 训练集 | 对照的评测集 |
|---|---|---|
| math | `dapo_math`（DAPO-Math-17K） | MATH 全部 5,000（含 MATH-500）、GSM8K test 1,319、AIME24 30、AIME25 30、AMC23 40、OlympiadBench 674 |
| code | `taco_verified`（TACO-verified） | HumanEval+ 164、MBPP+ 378、LiveCodeBench 131 |
| knowledge | `triviaqa_train`（TriviaQA train） | TriviaQA validation 全部 9,960、MMLU-Pro 全部 12,032、GPQA-Diamond 198、FreebaseQA 3,996 |
| instruction_following | `ifeval_train`（RLVR-IFeval） | IFEval 全部 541、IFBench 300、MT-Bench 80、AlpacaEval 805 |
| tool_calling | `sealtool_train`（Seal-Tools train） | Seal-Tools 全部 2,054、API-Bank 584、BFCL 4,696、MetaTool 5,327、T-Eval 4,358 |

## 去重规则

1. 从 `split_pool == pool` 开始（`make_splits.py` 已把去污染命中的题放进 `train_excluded`、内部重复放进 `train_duplicate`）。
2. 独立复查，命中任一条就移除：
   - 小写 + 空白规范化后与某道评测题完全相同；
   - 再去掉 LaTeX 标记、标点、空格后完全相同（抓格式差异）；
   - 13-gram 近似重复：与某一道评测题共享的 13-gram 占任一方 13-gram 的 ≥50%。
     IF 和工具两个领域里，出现在 ≥5 道评测题中的 13-gram 视为模板句（约束说明、调用格式说明），不计入。
   - 比较的文本：`prompt`，没有时取 `messages` 中的 user 轮；Seal-Tools 只比较 `task_instruction` 一行（与 `make_splits.py` 相同）。
3. 去掉训练集内部的重复题面（按第二条规范化，保留 id 最小的一条），这样训练和评测按题面也不重叠。
4. 排序后打乱，前 1,500 道作训练，接下来 500 道作评测。脚本断言两者按 id 和按题面都不重叠。

## 各领域数量

2026-11-21 起 `make_splits.py` 已做同样的去污染和去重（重复题放进 `train_duplicate`），所以这里的复查在当前池子上
不再移除任何题，只作为独立校验。

| 领域 | 原始 | 去污染（`train_excluded`） | 内部重复（`train_duplicate`） | 最终数据集 `pool.jsonl` | 训练 | 评测 |
|---|---|---|---|---|---|---|
| math | 17,382 | 1,541 | 1,103 | 14,738 | 1,500 | 500 |
| code | 10,806 | 0 | 15 | 10,791 | 1,500 | 500 |
| knowledge | 76,523 | 3,185 | 237 | 73,101 | 1,500 | 500 |
| instruction_following | 14,970 | 21 | 0 | 14,949 | 1,500 | 500 |
| tool_calling | 12,022 | 5 | 2 | 12,015 | 1,500 | 500 |

这套规则第一次（在旧池子上）运行时，比 `make_splits.py` 旧规则多抓出的重合题如下；它们现在已在 `train_excluded` 里。

- **math**（473）：MATH reserve 470（MATH-500 以外的 4,500 道 test 原题）、OlympiadBench 2、MATH-500 1。
  原来的 13-gram 检查只对比 test 池，规范化也只处理空白，所以漏了这些格式略有差异的原题。
- **knowledge**（387）：FreebaseQA 287、TriviaQA reserve 86、TriviaQA test 14。
  大多只差标点或引号，例如 `Who wrote the book A Brief History Of Time?` 与 FreebaseQA 的 ``Who wrote the book `A Brief History Of Time`?``。
- **instruction_following**（5）：都与 IFEval reserve 共用 "Give two different responses..." 这一约束句，训练题本身只有 "hi"、"你好" 之类几个字。
  这是模板造成的误报，但这几道题几乎没有指令内容，一并移除。
- **tool_calling**（2）：BFCL 1（`Tell me a joke`）、Seal-Tools reserve 1（同一句子任务嵌在一道多步评测题里）。
- **code**：0。
