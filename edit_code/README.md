# edit_code/：Paper A 自量化

计划见 `plan/PLAN.md` §3。尚未开始写代码。

环境：verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0 + cu130（`bash eval/hpc/setup_gpu_env.sh`）。每代的训练数据用 `data/scripts/draw.py` 抽取（配置 `data/draw/paper_a.yaml`）。计划中的模块：

- `diagnostics/`：预先算好的诊断表（逐模块 ΔKL、激活统计、行为敏感度、bytes）
- `quantize/`：按配置量化当前模型（GPTQ / AWQ / RTN），产出可被 vLLM 加载的权重
- `loop/`：一代的内层搜索（写程序 → evaluate → 历史）、验收、接任
- `feedback/`：L1 反馈（训练文本 KL + 五个领域信号切片）和 L2 验收；评测调用根目录 `eval/run_eval.py`
- `baselines/`：统一量化、KL 背包、EvoPress、GAMMA
