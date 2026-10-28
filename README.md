# ACMEC: HF 微调网络的优先连接与锁定效应
实验方案见 `data/Previous/HF微调网络_优先连接与锁定效应_实验方案.pdf`。

- 代码: `/home/caroyf/myproject/ACMEC` (本仓库)
- 数据/结果: `/net/spaces/jevans/klab/caroyf/data/ACMEC`,仓库内 `data/`、`results/` 为指向它的符号链接
- 环境: `conda activate hfpa && source activate.sh`(conda envs/pkgs 在数据盘,见 `~/.condarc`)
- 阶段脚本: `scripts/p0_*.py … p10_*.py`;Slurm 模板 `slurm/{cpu,gpu}.sbatch`
- 路径检查: `python -m hfpa.paths`;随机种子见 `configs/config.yaml`
