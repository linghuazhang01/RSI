# experiment_records/：实验记录

每次运行一个目录：`<日期>_<A|B>_<阶段>_<简述>/`，例如 `2026-11-25_A_P0_noise-band/`。目录里放：

- `config.yaml`：模型、预算、种子、数据池、代码版本
- `log.md`：做了什么、发生了什么（按时间）
- `results/`：`eval/run_eval.py` 的输出（每题打分 + summary.json）
- `summary.md`：结论、和预期的差别、下一步

只追加，不改旧记录；推翻旧结论时写新记录并互相链接。
