"""Builds notebooks/note4_results.ipynb (unexecuted). Executed afterwards by nbconvert in a Slurm job.

v2 (2026-11-26): narrative re-organised after the exploratory round (v2_explore.py: popularity split, intensive
margin, pioneers, growth ablations with refit, platform counterfactuals)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================================================================
md(r"""
# 笔记 4：What Drives Base-Model Choice（Section 5）的写作底稿（第二版）

这份笔记把 **Section 5** 需要的结果集中在一处：每个结论用哪张表、哪个数字支撑，怎么解释，放正文还是附录。

**第二版的变化（2026-11-26）：** 探索性实验已全量完成（`scripts/v2_explore.py`），据此重新确定了叙述逻辑。
- **新增 5.3 "流行由什么构成"：** 把流行拆开后，累积采纳（经典的马太效应）只占约四分之一。这是对"solid but boring"的主要回应。
- **5.4 增加集约边际：** 只看已被使用过的候选，流行仍占 94%。
- **增长消融、平台反事实、留存**移交给 Section 6 和 Section 7（§7）。
- **先行者分析删除**（§9）。
- §0.3 是全部结果的分配清单。

**口径：**
- **主规格 P5：** 首次采纳，(d+1) PPS 候选集，五组特征（流行、通用质量、需求匹配、自供给、习惯），按选择加权。
- **区间：** 50 次按作者的乘数 bootstrap。流行组内拆分用 20 次。

**代码：**
- 单元格已经运行过。重跑：`sbatch slurm/nb.sbatch notebooks/note4_results.ipynb`，约需 40 GB 内存；§4 读取 3,500 万行候选。
- 所有数字直接读自 `results/tables/v2/` 和 `processed/choices/v2_pps/v2_samples/`。

**目录**
0. 写作思路：主线、小节结构、全部结果的分配清单
1. 5.1 主结果：流行解释了几乎所有能解释的部分
2. 5.2 流行是不是隐藏的质量？
3. 5.3 流行由什么构成：大部分不是累积优势（新）
4. 5.4 适合：被看重、是开发者特有的，但很少起决定作用
5. 5.5 谁在跟随流行？
6. 5.6 稳健性（正文部分：样本选择）
7. 移交给 Section 6 和 Section 7 的结果：增长消融、平台反事实、留存
8. 附录 D：稳健性检验
9. 删除的结果：先行者
10. 表、图、宏与草稿修改
11. 审稿人可能的追问
""")

md(r"""
## 0. 写作思路

### 0.1 主线（一段话）
开发者选基座模型时：
- **流行解释了几乎所有能解释的部分**（92%，5.1）。
- **这不是隐藏的质量**（5.2）：即使把全部枢纽 FE 算作质量，流行仍占一半；质量完全相同的副本之间，胜出的是近期采纳更多的那个。
- **这里的"流行"也不是经典的马太效应**（5.3）：累积采纳只占增益的约四分之一。其余来自三部分：
  - "是否已被用过"的门槛（约五分之一）；
  - 近期动量（约四分之一）；
  - 平台可见度（约六分之一）。
- **适合被高度看重，但很少能把候选区分开**（5.4）：约 8 个候选同时匹配最好；即使只看已被使用过的候选，流行仍占 94%。
- **适合份额随账号的专业程度上升**（5.5）：典型开发者几乎只看流行。
- **网络层面的后果放在 Section 6：** 只靠累积优势，只能生成真实头部集中度的约三分之一，动量是必需的。

### 0.2 回应"solid but boring"
"流行占 92%"本身确实只是确认了马太效应，新意在于后两步：
1. **5.3 把"流行"拆开：** 文献中的优先连接只有累积 degree 一项，它在我们的数据里只占约四分之一，并且是强烈次线性的（$P\propto d^{0.34}$）。
2. **Section 6 的增长消融把拆分和集中联系起来：** 重新估计后，只保留累积项时，模拟网络的 top-10 份额是 0.07，真实是 0.22；去掉动量，份额减半。

这两点合起来，结论从"有马太效应"变成"**集中来自近期动量和使用门槛，而不是来自累积优势**"。这对 EC 读者是一个可检验、有含义的区分：前者更像羊群或注意力，后者更像从总数中学习质量（Section 7 讨论）。

### 0.3 全部结果的分配清单

**判断标准：**
- **正文：** 结果直接回答标题问题，或排除一个主要的替代解释，并且是稳定的（区间窄、留出集一致）。
- **附录：** 结果只检验某个设定选择，并且不改变结论。
- **Section 6 / 7：** 结果关于网络层面的后果或福利。
- **删除：** 结果自相矛盾或不稳定。

| # | 结果 | 位置 | 理由 | 本笔记 |
|---|---|---|---|---|
| 1 | 五组 Shapley 份额（样本内、留出集、区间） | **正文 5.1**，图 1 | 主结论 | §1 |
| 2 | 各组 LR 检验、留出集增益（×754） | **正文 5.1**（文字） | 区分"显著"与"重要" | §1 |
| 3 | 潜类别的选择份额与半合成检验 | 附录 D；正文一句 | 换单位复核，不改变结论 | §8 |
| 4 | 系数与边际替代率表 | 附录 D；正文引用 3 个读法 | 单位不可比，正文只用换算 | §3、§4 |
| 5 | FE 上界（流行仍占 51%） | **正文 5.2** | 排除"流行 = 质量" | §2 |
| 6 | 外部质量只解释 FE 的 7% | **正文 5.2** | 同上 | §2 |
| 7 | 同权重副本（动量胜出，累积无效） | **正文 5.2 结尾**，过渡到 5.3 | 质量固定的最干净证据 | §2 |
| 8 | 流行组内拆分，全量数据加区间 | **正文 5.3**，图 2 | 新发现，回应"boring" | §3 |
| 9 | 拆分从 10% 样本到全量的稳定性 | 附录 D | 只说明稳定 | §8 |
| 10 | 集合内指数变异 | 正文 5.4 一句；附录表 | 机制证据 | §4 |
| 11 | 并列数（7.9）与选择分类（7.5% / 51%） | **正文 5.4** | "匹配缩小范围，流行决定赢家" | §4 |
| 12 | 集约边际（d≥1、d≥10） | 正文 5.4 一句；附录表 | 排除两阶段解释 | §4 |
| 13 | donor 安慰剂 | **正文 5.4** 一段 | 排除"匹配 = 潮流" | §4 |
| 14 | 作者类型份额与异质性 LR | **正文 5.5**，图 3 | 新发现：谁在跟随流行 | §5 |
| 15 | 样本选择（U / D / P、作者等权、去掉头部 1%） | **正文 5.6**，`tab_samples` | 回应"少数作者"的核心质疑 | §6 |
| 16 | 扣除自身 degree 的修正 | 附录 D；正文一句 | 改变不到 1 个百分点 | §6 |
| 17 | 候选集设计（PPS / 分层 / 均匀） | 附录 D；正文一句 | 设定检验，原 `sec:robustq` 移入附录 | §8 |
| 18 | 需求信号子样本、文本门槛、旧卡片、10% 样本 | 附录 D | 设定检验 | §8 |
| 19 | 增长消融（重新估计） | **Section 6** | 网络层面的机制 | §7.1 |
| 20 | 集中度按效用项和作者类型分解 | Section 6 或附录 | 网络层面 | §7.2 |
| 21 | 平台反事实（匹配 ×2/×4、流行 ×0.5） | Section 6 结尾或 Section 7 | 政策含义 | §7.3 |
| 22 | 留存（按作者类型、福利） | Section 7 一段；附录 E | 后果，不是"什么驱动选择" | §7.4 |
| 23 | 恢复检验 | Section 4.6 | 方法验证 | — |
| 24 | 先行者 | **删除** | 回归和描述统计方向相反 | §9 |

### 0.4 建议的 Section 5 结构

| # | 小节标题（建议） | 一句话主张 | 关键数字 |
|---|---|---|---|
| 5.1 | Popularity accounts for almost all of what can be explained | 流行 92%，适合 5%，自供给 3%，习惯 0.3%，质量约 0；留出集相同 | 份额与区间、LR、×754 |
| 5.2 | Popularity is not hidden quality | FE 上界下流行仍占一半；外部质量只解释 FE 的 7%；同权重副本中动量胜出 | 51%、7%、0.84 对 −0.08 |
| 5.3 | What popularity is: mostly not cumulative advantage | 累积约 1/4、动量约 1/4、使用门槛约 1/5、可见度约 1/6；$P\propto d^{0.34}$ | 拆分表与区间 |
| 5.4 | Fit is valued, developer-specific, but rarely decisive | 匹配每单位价值很高，但约 8 个候选并列；已被使用的候选中流行仍占 94%；donor 进入后自己的匹配仍保留 | ×246、7.9、94%、4.5% |
| 5.5 | Who follows popularity? | 典型开发者流行 97%；适合随账号规模上升；自供给是组织行为 | 作者类型图、LR |
| 5.6 | Robustness | 所有替代口径都让适合变小或让习惯挤占流行；其余检验见附录 D | `tab_samples` |

**相对第一版的改动：**
- **5.2 和 5.3 的顺序：** 先排除质量（5.2），然后用副本结果"质量固定时动量胜出"自然引出"流行到底是什么"（5.3）。
- **系数的读法分散到各小节：** 外延边际和动量放 5.3，匹配的价值放 5.4，不再单独成节。
- **留存移出 Section 5：** 它回答的是"挑得好不好"，不是"什么驱动选择"。

### 0.5 写作上的几个原则
- **每个结论都给出"它不是什么"。**
  - 流行主导 ≠ 质量（5.2）；
  - 流行 ≠ 累积优势（5.3）；
  - 流行主导 ≠ 开发者不在乎适合（5.4）；
  - 匹配 ≠ 共同潮流（5.4）。
- **份额的单位始终是"似然增益的份额"。** 5.3 的拆分是对**全部**增益的份额（七个玩家的博弈），不是"流行内部的份额"。
- **动量与累积：** 两者的份额相当，并且在样本内和留出集上的排名互换。正文写"各约四分之一"，**不要写"动量最大"**。动量更重要的证据来自副本（5.2）和增长消融（Section 6）。
- **不要用"盲目跟风"之类的词。**
- **数字全部用宏。** §10 列出新增的宏。
""")

# =====================================================================================================================
code(r"""
import os, sys, re, json, glob, pickle, warnings
os.environ.setdefault("ACMEC_DATA_ROOT", "/net/spaces/jevans/klab/caroyf/data/ACMEC")
sys.path.insert(0, "/home/caroyf/myproject/ACMEC/src")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")
DATA = "/net/spaces/jevans/klab/caroyf/data/ACMEC"
TAB = f"{DATA}/results/tables/v2"
CH = f"{DATA}/processed/choices/v2_pps"
REPO = "/home/caroyf/myproject/ACMEC"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; GRAY = "#8a8984"; C5 = C + ["#8a5cc2"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "font.size": 9})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
show = display
tab = lambda name: pd.read_csv(f"{TAB}/{name}.csv")
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
P5 = pickle.load(open(f"{CH}/v2_samples/P5/fit.pkl", "rb"))
LAB = {"pop": "popularity", "qual": "general quality", "match": "demand match", "self": "self-supply", "habit": "habit"}
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(f"{REPO}/ec_adoption/tables/numbers.tex").read()))
mrs = tab("v2_mrs"); m1 = mrs[mrs.ref == "log_deg"].set_index("feature")[["dx", "multiplier", "mult_lo", "mult_hi"]]
cf = tab("v2s_P5_clogit").set_index("param")
print("main config:", P5["cfg"]); print(P5["info"])
""")

# =====================================================================================================================
md(r"""
## 1. 5.1 主结果：流行解释了几乎所有能解释的部分

### 1.1 五组 Shapley 份额（样本内与留出集）
下面的表和图是 5.1 的核心（图 1，或 Table `tab_shapley` 改为五组）。
- **样本内：** 份额与 50 次 bootstrap 的 95% 区间。
- **留出集：** 用训练集的估计预测最后 10% 的选择。一组特征只有能预测**未来**的选择，才能在这里得分。
""")

code(r"""
sh = tab("v2s_P5_shapley")
g5 = sh[(sh.variant == "no FE") & (sh.partition == "groups")]
pv = g5.pivot(index="group", columns="measure", values="share")
ci = g5[g5.measure == "in-sample"].set_index("group")[["lo", "hi"]]
T1 = pv.join(ci).loc[["popularity", "general quality", "demand match", "self-supply", "habit"]]
T1 = T1[["in-sample", "lo", "hi", "holdout"]]
show((T1 * 100).round(2).rename(columns=lambda c: c + " (%)"))
coarse = sh[(sh.variant == "no FE") & (sh.partition == "pop / fit / ...") & (sh.measure == "in-sample")].set_index("group")
show(coarse[["share", "lo", "hi"]].round(4))
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 2.8))
order = list(T1.index)
y = np.arange(len(order))
ax.barh(y - 0.18, T1["in-sample"] * 100, height=0.34, color=C[0], label="in-sample (95% bootstrap CI)")
ax.errorbar(T1["in-sample"] * 100, y - 0.18, xerr=[(T1["in-sample"] - T1.lo) * 100, (T1.hi - T1["in-sample"]) * 100],
            fmt="none", ecolor="black", lw=1, capsize=2)
ax.barh(y + 0.18, T1["holdout"] * 100, height=0.34, color=C[1], label="holdout")
for i, (a, b) in enumerate(zip(T1["in-sample"], T1["holdout"])):
    ax.text(max(a, b) * 100 + 1.5, i, f"{100 * a:.1f} / {100 * b:.1f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlim(0, 105)
ax.set_xlabel("share of the log-likelihood gain over uniform attachment (%)")
ax.set_title("Figure 1. What explains base-model choice (first adoptions, five groups)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.2 每一组都显著：似然比检验
从 32 次子集拟合可以直接得到"去掉一组"的似然比统计量：
$$LR_g = 2\,[\ell(\hat\theta_{\mathcal G}) - \ell(\hat\theta_{\mathcal G\setminus g})] = 2\,[G(\mathcal G) - G(\mathcal G\setminus g)],$$
自由度是该组的特征数。它在正文中的作用是**把"显著"与"重要"分开**：每一组都显著，所以问题不是"开发者是否回应适合"，而是"适合能解释多少选择"。
""")

code(r"""
fits, groups = P5["fits"], P5["groups"]
G = lambda S: fits[frozenset(S)]["ll"] - fits[frozenset()]["ll"]
allg = list(groups)
from scipy.stats import chi2
lr = []
for g in allg:
    st_ = 2 * (G(allg) - G([h for h in allg if h != g]))
    df = len(groups[g])
    lr.append(dict(group=LAB[g], features=df, LR=st_, p=chi2.sf(st_, df) if df else np.nan))
show(pd.DataFrame(lr).round(2))
info = P5["info"]; h = P5["nofe"]["hold"]
print(f"holdout: gain {h['mean_ll'] - P5['u0h']:.3f} nats/choice (x{np.exp(h['mean_ll'] - P5['u0h']):,.0f}), "
      f"top-1 {pct(h['top1'])}, top-10 {pct(h['top10'])}; pseudo-R2 in-sample {1 - P5['nofe']['ll'] / P5['u0']:.3f}")
""")

md(r"""
**写作要点（5.1）：**
1. **第一段直接给主结论，** 配图 1：流行 91.6%（91.1–91.9），适合 5.2%，自供给 2.9%，习惯 0.3%，质量 < 0.01%。
2. **紧接着给留出集。** 份额几乎不变，说明排序不是样本内拟合的产物。
3. **用一句话把"显著"和"重要"分开**（LR 检验）。
4. **潜类别结果一句话即可：** "按选择计，92% 的选择归于流行模式"，细节放附录 D（§8.4）。
5. **在这里第一次解释自供给：** 它是"选自己的模型"，主要是组织行为（5.5）。
6. **结尾提出 5.2 的问题：** 流行是不是质量的代理？
""")

# =====================================================================================================================
md(r"""
## 2. 5.2 流行是不是隐藏的质量？

benchmark 分数只覆盖很少的候选，所以"质量份额约 0"可能只是没测到质量。下面三种证据的强度依次递增。

### 2.1 上界：把枢纽 FE 全部算作质量
加入 500 个枢纽的固定效应（Section 4.5 定义 4），作为 Shapley 的额外玩家。FE 吸收枢纽所有不变的属性。把它全部算作质量，是对质量**最宽松**的解读。
""")

code(r"""
show(tab("v2_bounds").round(4))
fit_ = tab("v2_pps_clogit_fit").set_index("model")
print(f"holdout gain: no FE {fit_.loc['full', 'hold_gain']:.3f}, with FE {fit_.loc['full + FE', 'hold_gain']:.3f} nats/choice")
fe5 = sh[(sh.variant == "FE as a player") & (sh.partition == "pop / fit / ...")].pivot(index="group", columns="measure", values="share")
show((fe5 * 100).round(2))
""")

md(r"""
### 2.2 外部质量能解释 FE 的多少
用枢纽的外部质量（Open LLM Leaderboard v2 平均分、Arena 评分）回归 $\hat\phi_h$，控制参数量和再分发账号，并做噪声校正（Section 4.5 的噪声引理）。
""")

code(r"""
qe = tab("v2_quality_ext")
show(qe[[c for c in ["source", "controls", "hubs", "share_of_fe", "r2_noise_corrected", "quality_part_share"] if c in qe]].round(3))
""")

md(r"""
### 2.3 质量固定时：同权重副本
有 638 次首次采纳满足两个条件：候选集中同时有两个**同一份权重**的副本（权重哈希相同，由不同账号发布），并且其中一个被选中。质量在构造上完全相同。用两个副本的特征差做条件 logit，看是什么让一个副本胜出。
""")

code(r"""
show(tab("v2_identical_pairs").round(4))
""")

md(r"""
**怎么读：**
- **上界：** 即使把全部 FE 算作质量，流行仍占约 51%（样本内）。FE 几乎不改善样本外预测（6.625 → 6.632 nats）。
- **外部质量：** 只能解释 FE 方差的约 7%（118 个被覆盖的枢纽，噪声校正后）。按此拆分，适合份额只升到 7.0%。
- **同权重副本：**
  - 胜出的是近 30 天采纳更多的副本（系数 0.84，标准误 0.08），而不是总采纳更多的副本（−0.08，标准误 0.10）；
  - 被选副本的总 degree 更高的比例只有 48.6%，相当于抛硬币。

**写作要点（5.2）：**
- **三层证据的顺序：** 上界 → 外部质量 → 质量固定。
- **以副本结果收尾，并用它过渡到 5.3：** "质量固定时，起作用的不是总采纳，而是近期采纳。这提示'流行'这个组内部的结构很重要。"
- **待办：** benchmark 只覆盖 118/500 个枢纽，代表性需要讨论。
""")

# =====================================================================================================================
md(r"""
## 3. 5.3 流行由什么构成：大部分不是累积优势（新）

### 3.1 设计
把流行组拆成六个玩家，加上其余四组合并成的"非流行"玩家，一共七个玩家，做 $2^7=128$ 次子集拟合的 Shapley 分解。全部首次采纳，20 次按作者的乘数 bootstrap。

| 玩家 | 特征 | 含义 |
|---|---|---|
| 是否被采纳过（extensive） | `has_deg` | 有没有人用过：使用门槛 |
| 累积（cumulative） | `log_deg` | 总采纳数：经典优先连接 |
| 动量（momentum） | `lrec30` | 近 30 天的采纳 |
| 生命周期（lifecycle） | `log_age`、`legacy` | 年龄与旧代模型 |
| 可见度（visibility） | `major_org`、`redistributor`、`lquant`、缺失标记、`lib_*` | 品牌、再分发、量化版本、库 |
| 局部（local） | `lco2`、`ds_couse` | 两跳共同使用者、数据集共同使用 |
| 非流行 | 质量、匹配、自供给、习惯 | 合并为一个玩家 |
""")

code(r"""
ps = tab("v2x_popsplit_full").set_index("group")
PSL = {"extensive": "used at all (has_deg)", "cumulative": "cumulative adoptions (log_deg)",
       "momentum": "momentum, 30 days (lrec30)", "visibility": "visibility (org, redistribution, quant, library)",
       "local": "local (co-users, dataset co-use)", "lifecycle": "lifecycle (age, legacy)", "nonpop": "all non-popularity groups"}
ps = ps.loc[["extensive", "cumulative", "momentum", "visibility", "local", "lifecycle", "nonpop"]]
show((ps[["share", "lo", "hi", "hold_share"]] * 100).round(2).rename(index=PSL))
print(f"popularity total in the 7-player game: {100 * (1 - ps.loc['nonpop', 'share']):.1f}% "
      f"(five-group game: {100 * T1.loc['popularity', 'in-sample']:.1f}%); bootstrap reps {int(ps.reps.iloc[0])}")
for f in ["has_deg", "log_deg", "lrec30", "major_org", "redistributor"]:
    r = m1.loc[f] if f in m1.index else None
    extra = f"; multiplier on (1+d) {r.multiplier:,.1f} [{r.mult_lo:,.1f}, {r.mult_hi:,.1f}] (dx = {r.dx:.3g})" if r is not None else ""
    print(f"{f:14s} coef {cf.loc[f, 'coef']:.3f} (se {cf.loc[f, 'se_cl']:.3f}){extra}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.5, 3))
yy = np.arange(len(ps))
cols = [C[0]] * 6 + [GRAY]
ax.barh(yy, ps.share * 100, color=cols, height=0.6)
ax.errorbar(ps.share * 100, yy, xerr=[(ps.share - ps.lo) * 100, (ps.hi - ps.share) * 100], fmt="none", ecolor="black",
            lw=1, capsize=2)
ax.scatter(ps.hold_share * 100, yy, marker="D", s=22, color=C[1], zorder=3, label="holdout")
for i, v in enumerate(ps.share):
    ax.text(max(v, ps.hold_share.iloc[i]) * 100 + 0.8, i, f"{100 * v:.1f}", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels([PSL[g] for g in ps.index]); ax.invert_yaxis(); ax.set_xlim(0, 30)
ax.set_xlabel("share of the total log-likelihood gain (%), bars = in-sample with 95% bootstrap CI")
ax.set_title("Figure 2. Cumulative adoptions are only a quarter of what popularity explains", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

md(r"""
### 3.2 怎么读
**1. 累积优势只占约四分之一。**
- 经典优先连接（总采纳数）只占全部增益的 23.4%（23.2–23.6）。
- 主模型中 `log_deg` 的系数是 0.34，即 $P\propto d^{0.34}$，远低于 BA 模型的 1。单独估计累积项时，系数也只有 0.70（§7.1）。

**2. 另外三块同样大。**
- **近期动量：** 24.9%（24.5–25.2）。
- **使用门槛：** "是否被采纳过"占 20.3%。`has_deg` 的系数是 4.64，从"没人用过"到"有人用过"，概率约乘以 100 倍。
- **平台可见度：** 品牌、再分发账号、量化版本、库，合计 17.2%。

**3. 动量和累积哪个更大，这里分不出来。**
- 样本内动量略高（24.9 对 23.4，区间不重叠），留出集上反过来（23.4 对 24.5）。
- 正文只能写"各约四分之一"。动量更重要的证据来自另外两处：
  - 质量固定时（5.2 副本），只有动量起作用；
  - 生成集中时（Section 6），去掉动量后集中减半，只留累积时集中只有真实的三分之一。

**4. 关于博弈划分的脚注。** 七个玩家的博弈中，流行合计 93.2%，五组博弈中是 91.6%。Shapley 值对玩家的拆分不是不变的：把一个玩家拆细，会改变它与其他玩家的交互项的分配。所以：
- 正文报告拆分时，用"占全部增益的份额"；
- 加一个脚注说明这 1.6 个百分点的差异。

**5. 稳定性。** 从 10% 样本到全量，每个子组的份额变化不超过 1 个百分点（附录 D，§8.3）。

### 3.3 写作要点（5.3）
- **开头承接 5.2：** "5.2 显示质量固定时近期采纳起作用而总采纳不起作用。这一节问：'流行'这个组到底由什么构成？"
- **主体：** 图 2 加一段读法，强调"累积优势只是四分之一"和"$d^{0.34}$"。
- **EC 含义（一句，详细讨论放 Section 7）：** 从总采纳中学习质量（社会学习）预测的是累积项起作用；动量和门槛更像注意力、羊群或"可用性证明"。
- **结尾引出 Section 6：** "这些成分中，哪个生成了观察到的集中？"
- **不要做的：** 不要说"动量最重要"；不要把拆分说成因果。
""")

# =====================================================================================================================
md(r"""
## 4. 5.4 适合：被看重、是开发者特有的，但很少起决定作用

### 4.1 问题
匹配每单位的价值很高：候选的任务出现在作者历史中（`task_hist`），相当于 degree 乘以约 246 倍。但它只解释 5% 的增益。

这两点并不矛盾。一组特征能解释多少选择，取决于它的**效用指数在候选集内变异多大**（Section 4.4 的第 (iii) 点）。
""")

code(r"""
for f in ["task_hist", "lang_hist", "txt_sim"]:
    if f in m1.index:
        r = m1.loc[f]
        print(f"{f:12s} coef {cf.loc[f, 'coef']:.3f}; multiplier on (1+d): {r.multiplier:,.1f}  [{r.mult_lo:,.1f}, {r.mult_hi:,.1f}]  (dx = {r.dx:.3g})")
""")

md(r"""
### 4.2 证据一：各组指数在候选集内的变异
对训练集的每次选择，计算每组的效用指数 $I_{g,kl}=\hat\theta_g^\top x_{kl,g}$ 在候选集内的标准差和极差。
""")

code(r"""
from hfpa.v2data import load_rows, split
rows, ch = load_rows("v2_pps", "planA")
tr, _ = split(rows, ch)
tr = tr.sort_values("choice_id", kind="stable").reset_index(drop=True)
feats, b = P5["feats"], P5["nofe"]["coef"]
X = tr[feats].to_numpy(np.float64)
cid = tr.choice_id.to_numpy(); st = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]; ln = np.diff(np.r_[st, len(cid)])
I = {g: X[:, [feats.index(f) for f in v]] @ b[[feats.index(f) for f in v]] for g, v in groups.items() if v}
chosen = tr.chosen.to_numpy() == 1
rows_out = []
for g, v in I.items():
    mu = np.add.reduceat(v, st) / ln
    sd = np.sqrt(np.maximum(np.add.reduceat(v * v, st) / ln - mu ** 2, 0))
    rng_ = np.maximum.reduceat(v, st) - np.minimum.reduceat(v, st)
    adv = v[chosen] - mu                      # chosen minus set mean (choices are in the same order)
    rows_out.append(dict(group=LAB[g], within_sd_median=np.median(sd), within_sd_mean=sd.mean(),
                         within_range_median=np.median(rng_), chosen_minus_mean=adv.mean(),
                         share_sets_constant=(sd < 1e-9).mean()))
VAR = pd.DataFrame(rows_out).set_index("group")
print(f"{len(st):,} training choices; {len(tr):,} rows")
show(VAR.round(3))
del X
""")

md(r"""
### 4.3 证据二：多少候选同时"匹配最好"，以及被选者是哪一类
在每次选择的候选中（只看已被使用过的候选，即 $d>0$），判断被选的父模型是哪一类：
- 匹配指数最高的（"适合型"）；
- 流行指数最高的（"流行型"）；
- 两者都是，或者都不是。

并列时随机打破，报告期望份额。
""")

code(r"""
show(tab("v2_classify").round(4))
""")

md(r"""
### 4.4 证据三：集约边际，即只在已被使用过的候选之间
**替代解释（两阶段）：** 流行只决定一个候选能否进入考虑范围（有没有人用过），在已被使用过的候选之间，由适合决定。

**检验：** 把候选集限制为 $d\ge1$ 或 $d\ge10$ 的候选，并且只保留被选者也满足这个条件的选择，重新做五组分解。
""")

code(r"""
mg = tab("v2x_margin"); mg = mg[mg.frac == mg.frac.max()]
show(mg.pivot_table(index=["min_deg", "kept", "set_size"], columns="measure", values=["popularity", "fit", "self", "habit"]).round(4))
""")

md(r"""
**怎么读：** 两阶段解释不成立。
- 只看 $d\ge1$ 的候选（保留 89% 的选择，候选集平均 46 个）：流行仍占 94.3%，适合 5.4%。
- 只看 $d\ge10$ 的候选（保留 79%）：流行 94.0%，适合 5.6%，与全集合几乎相同。

即使在已被使用过的候选之间，流行仍然决定赢家。

### 4.5 证据四：适合是开发者特有的需求吗？（donor 安慰剂）
**替代解释：** 匹配份额可能只是共同的潮流。

**检验：** 把每个开发者的需求换成同月同模态另一位作者的需求，重算匹配特征。
""")

code(r"""
show(tab("v2_placebo").round(4))
show(tab("v2_placebo_joint").round(4))
""")

md(r"""
**怎么读：**
- 单独使用 donor 的需求时，匹配份额为 3.1%，说明共同潮流存在。
- 自己的需求和 donor 的需求同时进入时，自己的匹配保留 4.5%，donor 只有 2.4%。
- donor 挤占的是**流行**的份额，而不是自己的匹配。

### 4.6 写作要点（5.4）
**顺序：**
1. 匹配的价值很高（×246）；
2. 但匹配很少能区分候选：集合内变异小，平均 7.9 个候选并列匹配最好；
3. 核心句："**fit narrows the field; popularity picks the winner**"；
4. 集约边际一句，排除两阶段解释，表放附录 D；
5. donor 安慰剂一段：剩下的适合是开发者特有的。

**措辞：** 被选者恰好是集合中最流行者的选择只占 7.5%，"都不是"的占 51%。所以"跟随流行"是概率意义上的，不是"人人选最热门"。
""")

# =====================================================================================================================
md(r"""
## 5. 5.5 谁在跟随流行？

六类作者分别估计完整模型和 32 次子集拟合，各做 50 次 bootstrap。跨类型比较用份额，不用系数，因为 logit 的尺度在类型之间可以不同。
""")

code(r"""
R = tab("v2s_robustness").set_index("config")
types = ["indiv_light", "indiv_mid", "indiv_heavy", "org_other", "org_redistributor", "org_major"]
TL = {"indiv_light": "individuals <10 models", "indiv_mid": "individuals 10-99", "indiv_heavy": "individuals >=100",
      "org_other": "other organizations", "org_redistributor": "re-distribution accounts", "org_major": "major organizations"}
cols = ["n_train", "authors", "popularity", "popularity lo", "popularity hi", "fit (quality + match)",
        "fit (quality + match) lo", "fit (quality + match) hi", "self-supply", "habit", "ratio fit / popularity"]
TT = R.loc[[f"P5_{t}" for t in types], cols].rename(index=lambda s: TL[s[3:]])
share_first = TT.n_train / TT.n_train.sum()
TT.insert(2, "share of first adoptions", share_first)
show(TT.round(4))
show(tab("v2s_hetero"))
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 3))
parts = [("popularity", C[0]), ("fit (quality + match)", C[1]), ("self-supply", C[2]), ("habit", C[3])]
left = np.zeros(len(TT))
yy = np.arange(len(TT))
for name, col in parts:
    v = TT[name].to_numpy() * 100
    ax.barh(yy, v, left=left, color=col, height=0.6, label=name, edgecolor="white", linewidth=1)
    left += v
for i, (n, s) in enumerate(zip(TT.n_train, TT["share of first adoptions"])):
    lab_s = f"{100 * s:.0f}%" if s >= 0.005 else f"{100 * s:.1f}%"
    ax.text(101, i, f"n = {int(n):,} ({lab_s})", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(TT.index); ax.invert_yaxis(); ax.set_xlim(0, 125)
ax.set_xlabel("share of the log-likelihood gain (%), first adoptions")
ax.set_title("Figure 3. Who follows popularity: shares by author type", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.42, -0.42))
plt.tight_layout(); plt.show()
""")

md(r"""
**怎么读：**
- **典型开发者几乎完全跟随流行：** 轻度个人占首次采纳的一半，流行占 97.1%，适合 2.1%。
- **适合份额随账号的规模和专业程度上升：** 从 7.8%（中等个人）到 17.3%（大机构）。
- **自供给是组织行为：** 大机构 21.5%，轻度个人 0.8%。
- **合并估计被显著拒绝：** $LR=38{,}158$，自由度 130。
- **与 5.1 的关系：** 主结果中 5.2% 的适合份额，主要来自少数高产账号和组织。

**写作要点（5.5）：**
- 配图 3。
- **措辞：** "fit matters more for experienced and organizational developers"。
- **样本量：** 大机构只有 254 次选择，提到时注明。
- **再分发账号是按行为定义的，** 需要在 Section 3 或附录 A 说明。
- 图中 "(0%)" 的标签已改为保留一位小数。
""")

# =====================================================================================================================
md(r"""
## 6. 5.6 稳健性（正文部分：样本选择）

正文只保留样本选择这一项，因为它回应"少数作者"这个最可能的质疑。其余检验放附录 D（§8），正文一句话列出。

判断规则是运行前确定的（Section 4.7）：
- 流行份额与主结果相差不超过 10 个百分点；
- 适合/流行比值在主结果的 1/3 到 3 倍之间。
""")

code(r"""
main_cfg = ["P5", "P5top", "P5a", "D5", "D5raw", "D5a", "D5top", "U5"]
NM = {"P5": "first adoptions (main)", "P5top": "first adoptions, top 1% authors dropped",
      "P5a": "first adoptions, authors weighted equally", "D5": "all 24h decisions",
      "D5raw": "all decisions, no self-degree correction", "D5a": "all decisions, authors weighted equally",
      "D5top": "all decisions, top 1% dropped", "U5": "all uploads"}
RS = R.loc[[c for c in main_cfg if c in R.index], ["n_train", "authors", "popularity", "popularity lo", "popularity hi",
            "fit (quality + match)", "self-supply", "habit", "ratio fit / popularity", "d_pop_pts", "ratio_vs_main",
            "robust", "reps"]].rename(index=NM)
show(RS.round(4))
""")

md(r"""
**怎么读：**
- **按规则，只有"去掉前 1% 作者"通过。** 但所有偏离的方向一致：
  - 加入重复选择后，习惯挤占流行（全部决策中 21.9%，全部上传中 45.4%）；
  - 作者等权或去掉头部后，适合进一步变小（1.8%、3.5%）。
- **没有任何口径让适合比主结果更重要。** 正文应写："流行主导在所有口径下都成立；主结果的适合份额是各口径中的偏高端。"
- **扣除自身 degree 的修正**改变份额不到 1 个百分点（D5 与 D5raw），正文一句。

**写作要点（5.6）：**
- 第一段：样本选择，配 `tab_samples` 的上半部分，并如实报告规则的结果。
- 第二段（一句）："候选集设计、需求信号子样本、文本门槛、旧卡片、10% 样本、潜类别的结果见附录 D，都不改变结论。"
""")

# =====================================================================================================================
md(r"""
## 7. 移交给 Section 6 和 Section 7 的结果

这些结果不回答"什么驱动选择"，而是回答"这种选择方式在网络层面和福利上意味着什么"。下面整理它们的现状和用法，供写 Section 6/7 时使用。

### 7.1 增长消融：哪个成分生成了集中（→ Section 6，与 5.3 呼应）
**设计：**
1. **重新估计：** 在全部首次采纳上，重新估计去掉某些流行项的五组模型，让其余系数调整。
2. **模拟：** 用各模型的流行系数，在全市场做带反馈的增长模拟（`v2_networks.SF`），5 个种子。真实的事件序列（谁、何时、哪个模态）保持不变，只重抽父模型。

第一轮小规模实验没有重新估计，只把系数设为 0。那样夸大了动量的作用，并且把年龄衰减的方向弄反了，所以这里只报告重新估计的版本。
""")

code(r"""
show(tab("v2x_refit").round(4))
sr = tab("v2x_sfrefit").set_index("variant")
show(sr[["top10_mean", "top10_min", "top10_max", "eff_parents_mean", "gini_mean", "turnover_mean", "n_parents_mean"]].round(3))
""")

md(r"""
**怎么读：**
- **完整流行模型** 再现了真实的 top-10 份额（0.237，5 个种子 0.21–0.27，真实 0.219）和头部更替率（0.48 对 0.46）。
- **只留累积项：** top-10 份额只有 0.071。单独估计时累积项的系数是 0.70，仍然次线性，**只能生成真实头部集中度的约三分之一**。
- **去掉动量：** top-10 份额减半（0.119，区间与完整模型不重叠），尽管累积系数从 0.34 升到 0.79 来补位。动量是生成头部集中的必要成分。
- **去掉年龄：** top-10 份额是 0.175，更替率上升到 0.53。小规模（不重新估计）时，去掉年龄会让集中升高；重新估计后方向相反，因为累积系数降到 0.16、动量升到 0.77。**"年龄衰减压住集中"这个说法不成立，不要写。**
- **似然损失：** 去掉动量损失 17,915 nats，去掉年龄损失 9,425 nats，只留累积损失 40,778 nats。

**写进 Section 6 之前必须处理的问题：** 这里的模拟器与 Section 6 现有的增长网络（`v2_grow.py`）不同，两者的集中水平不一样：

| | `v2_networks.SF`（本节） | `v2_grow.py`（Section 6 现有） |
|---|---|---|
| 候选 | 全市场 | 在模拟 degree 上抽 PPS 集合 |
| 特征 | 只有可对任意节点计算的流行项 | 全部 26 个特征随模拟状态重算 |
| degree 的增长 | 只有首次采纳 | 全部 42.3 万次主选择（重复选择跟随首次选择） |
| SF 的 top-10 份额 | 0.24（与真实相近） | 0.39（过度集中） |
| SF 的有效父模型数 | 2,647（主体太分散） | 439 |

**可能的原因（推测，未验证）：** `v2_grow` 把重复选择也计入 degree，degree 增长约快 3 倍，正反馈更强。

**建议：**
- 在 `v2_grow` 中用重新估计的系数重跑两个关键消融（"去掉动量"和"只留累积"）。如果排序相同，D 进 Section 6 正文；否则放附录，正文只写方向。这是一个小作业，需要你确认后再提交。
- 无论哪种情况，都要写明本节模拟器的主体集中度不够（有效父模型约 2,600，真实 839）。
""")

code(r"""
gr = tab("v2_grow").set_index("network")
show(gr[["top10_mean", "eff_parents_mean", "gini_mean", "turnover_mean", "n_parents_mean"]].round(3))
""")

md(r"""
### 7.2 集中度按效用项和作者类型分解（→ Section 6 或附录）
**设计：** 在真实候选集上做重要性重抽样，不带反馈，比较只用部分效用项时网络有多集中。

**主要结论：**
- 流行项单独就能生成真实的 top-10 份额（0.225 对 0.219）；
- 加入匹配，集中几乎不变；
- 自供给使网络分散（有效父模型 1,118 → 1,365）。
""")

code(r"""
cz = tab("v2s_conc_P5")
show(cz[cz.model.isin(["REAL", "pop", "pop + match", "pop + self", "full (no FE)"])][["model", "top10", "eff_parents", "gini", "n_parents"]].round(3))
""")

md(r"""
### 7.3 平台反事实（→ Section 6 结尾或 Section 7）
**设计：** 把匹配组或流行组的效用乘以一个系数，在真实候选集上重算选择概率，计算网络集中度和留存变化。
- 留存变化 $\approx\beta\cdot\Delta I_{\text{match}}\cdot\bar p(1-\bar p)$，其中 $\beta=0.22$ 是个人作者的作者 FE 留存系数。
- 区间来自 50 组 bootstrap 系数。
""")

code(r"""
pf = tab("v2x_platform_full").set_index("variant")
show(pf[["top10", "top10_lo", "top10_hi", "eff", "eff_lo", "eff_hi", "chosen_match", "d_retention_pp", "d_retention_pp_lo", "d_retention_pp_hi"]].round(4))
""")

md(r"""
**怎么读：**
- **匹配权重 ×2：** 留存 +0.38 个百分点（0.36–0.64），集中度不变（top-10 份额和有效父模型数的区间都与基准重叠）。×4 时 +0.72（0.62–1.26）。
- **流行权重 ×0.5：** 市场接近均匀（top-10 份额 0.008），留存 −0.39 个百分点。
- **一句话：** 推荐时更看重适合，能提升留存而不改变集中；压低流行会打散市场，也会降低留存。

**注意事项：**
- 这是局部均衡，没有反馈；留存用的是线性近似。
- 比较对象是模型自己的基准（有效父模型 1,807），不是真实网络（839）。
- 流行 ×0.5 时"是否被采纳过"的系数也一起减半，情景很极端。

**建议：** 放在 Section 7 的"平台设计含义"中写一段，表放附录。不要放在 Section 5。
""")

md(r"""
### 7.4 留存：匹配好的选择是否更持久？（→ Section 7 一段；附录 E）
**方法：** 作者 FE 的条件 logit，因变量是下一次决策是否仍用同一父模型（笔记 3 §10）。
""")

code(r"""
rb = tab("v2s_retention_bytype")
fe_ = rb[(rb.model == "conditional logit, author FE") & rb["var"].isin(["I_pop", "I_match", "I_self"])]
show(fe_.pivot(index="stratum", columns="var", values="coef").round(3))
show(fe_.pivot(index="stratum", columns="var", values="z").round(1))
show(tab("v2s_retention_welfare").round(4))
""")

md(r"""
**怎么读：**
- **对个人作者，** 同一作者内部匹配好的父模型更常被继续使用（轻度个人 +0.47）。
- **对高产账号和组织，** 匹配不起作用。
- **福利：** 个人作者如果改选匹配最好的候选，留存概率提高 2.4 个百分点（基线的 8.6%）。

**建议：**
- 留存回答的是"挑得好不好"，不是"什么驱动选择"，所以移出 Section 5。
- 在 Section 7 中作为"跟随流行是否有代价"的证据写一段，与 7.3 的平台反事实放在一起。
- 表放附录 E，并删去旧草稿中"匹配系数为负"的表述。
""")

# =====================================================================================================================
md(r"""
## 8. 附录 D：稳健性检验

正文 5.6 用一句话指向这里。每一项给出：检验什么、结果、结论是否改变。

### 8.1 候选集设计（原 `sec:robustq`，移入附录）
""")

code(r"""
rows_q = []
for spec in ["v2_pps", "v2_strat", "v2_unif"]:
    bq = tab(f"{spec}_shapley_boot"); bq = bq[bq.partition == "pop / fit / inertia"].set_index("group")
    rows_q.append(dict(design=spec, popularity=bq.share["popularity"], fit=bq.share["fit (quality + match)"],
                       inertia=bq.share["inertia"], ratio=bq.share["ratio fit / popularity"],
                       ratio_lo=bq.lo["ratio fit / popularity"], ratio_hi=bq.hi["ratio fit / popularity"]))
show(pd.DataFrame(rows_q).set_index("design").round(4))
""")

md(r"""
- **PPS 与分层设计一致。**
- **均匀设计的适合份额更高：** 均匀候选集几乎不含热门竞争者，选择变成了"已知 vs 未知"的比较（笔记 3 §2.8）。
- **四组口径：** 这里的惯性等于自供给加习惯，附录中注明即可。

### 8.2 需求信号、文本门槛、旧卡片、10% 样本
""")

code(r"""
sg = tab("v2_signal_samples"); show(sg[sg.partition == "pop / fit / inertia"].round(4))
show(tab("v2_text_threshold").round(4))
show(tab("v2_old_cards").round(4))
v10 = tab("v2_validate_10pct")
print(f"10% vs full: {int(v10.in_10pct_ci.sum())} of {len(v10)} coefficients inside the 10% CI; median SE ratio {v10.se_ratio.median():.2f}")
""")

md(r"""
- **需求信号：** 在带有需求信号的 67% 的选择上，匹配份额是 8.3%；按作者经验和机构重新加权后是 6.4%。主结果在这个意义上是适合份额的下界。
- **文本门槛：** 最低词数从 5 提高到 50，匹配份额不变（5.2%–5.3%）。
- **旧卡片：** 枢纽被大量采用前的卡片版本与当前版本的相关系数是 0.51，结论方向相同。
- **10% 样本与全量：** 系数基本一致。

### 8.3 流行组内拆分的稳定性（支撑 5.3）
""")

code(r"""
pz = tab("v2x_popsplit")
show(pz[pz.measure == "ll"].groupby("frac")[["extensive", "cumulative", "momentum", "lifecycle", "visibility", "local", "nonpop"]].agg(["mean", "min", "max"]).round(3))
""")

md(r"""
从 10% 样本到全量，每个子组的份额变化都在约 1 个百分点之内。动量与累积的排名在样本内稳定，在留出集上反过来（§3.2）。

### 8.4 潜类别的选择份额与半合成检验（支撑 5.1 的一句话）
""")

code(r"""
mx = tab("v2_mixture_planA").set_index("quantity").value
mix = pd.Series({k.replace("share_", ""): float(mx[k]) for k in mx.index if k.startswith("share_")})
show((mix * 100).round(2).rename("share of choices (%)").to_frame())
ms = tab("v2_mixsim_planA")
show(ms.groupby(["dgp", "mode"])[["realised_share", "est_share", "abs_error"]].mean().round(4))
""")

md(r"""
### 8.5 其他放附录的表
- **系数与边际替代率表**（`tab_coef`）：正文只用其中三个读法（`has_deg`、`lrec30`、`task_hist`），以及 $d^{0.34}$。
- **集合内指数变异表**（§4.2）。
- **集约边际表**（§4.4）。
- **各组 LR 检验表**（§1.2），如果正文只写文字的话。
""")

# =====================================================================================================================
md(r"""
## 9. 删除的结果：先行者

**原问题：** 早期采纳者（父模型的前 10 次首次采纳）是不是更专业、更看重适合，然后带动了后来者的流行？
""")

code(r"""
pn = tab("v2x_pioneers"); show(pn[pn.frac == pn.frac.max()].round(3))
pp = tab("v2x_pioneers_profile"); show(pp[pp.frac == pp.frac.max()].round(3))
""")

md(r"""
**为什么删除：**
- **两组证据方向相反：**
  - 回归中，早期专家比例的系数为正（+1.9，标准误 0.2），即专家早期采纳的父模型后来更成功；
  - 但描述统计中，后来最成功的 10% 父模型，早期专家比例反而更低（0.25 对 0.32）。
- **专家比例随采纳次序呈 U 型。**
- **样本已经是全部：** 只有 1,006 个父模型有至少 10 次首次采纳，没有更大的样本可以澄清。
- **和主线的关系很弱：** 它回答的是"谁带动流行"，而不是"什么驱动选择"。

**做法：** 不在论文中报告。如果审稿人问"先行者是否不同"，可以在回复信中引用这两个数字。
""")

# =====================================================================================================================
md(r"""
## 10. 表、图、宏与草稿修改

### 10.1 正文的表和图

| 编号 | 内容 | 小节 | 来源 | 状态 |
|---|---|---|---|---|
| 图 1 | 五组份额（样本内 vs 留出集） | 5.1 | §1 | 需要从本笔记导出到 `ec_adoption/figures/` |
| 表 | 质量上界三档（`tab_bounds`） | 5.2 | §2 | 已有 |
| 图 2 | 流行组内拆分 | 5.3 | §3 | **新**，需要导出 |
| 图 3 | 作者类型堆叠图 | 5.5 | §5 | 需要导出 |
| 表 | 样本选择（`tab_samples` 上半部分） | 5.6 | §6 | 已有 |

`tab_shapley` 可以由图 1 替代，表移入附录。

### 10.2 附录 D 的表
`tab_coef`、集合内变异、集约边际、候选集设计（`tab_q`）、需求信号、文本门槛、旧卡片、10% 样本、潜类别、拆分稳定性。

### 10.3 需要新增的宏
由 `paper_tables.py` 生成，尚未实现：
""")

code(r"""
new = {
    "psExt / psCum / psMom / psVis / psLoc / psLife / psNonpop": " / ".join(f"{100 * ps.loc[g, 'share']:.1f}" for g in ps.index),
    "psCumLo, psCumHi, psMomLo, psMomHi": f"{100 * ps.loc['cumulative', 'lo']:.1f}-{100 * ps.loc['cumulative', 'hi']:.1f}; {100 * ps.loc['momentum', 'lo']:.1f}-{100 * ps.loc['momentum', 'hi']:.1f}",
    "psPopTotal": f"{100 * (1 - ps.loc['nonpop', 'share']):.1f} (7-player game, footnote)",
    "alphaDeg": f"{cf.loc['log_deg', 'coef']:.2f} (P proportional to d^alpha)",
    "marginOnePop / marginOneFit / marginTenPop / marginTenFit": " / ".join(
        f"{100 * mg[(mg.min_deg == d) & (mg.measure == 'll')][c].iloc[0]:.1f}" for d in [1, 10] for c in ["popularity", "fit"]),
    "lrSelf / lrHabit / lrMatch5 / lrPop5": "five-group LR statistics (§1.2)",
    "mixPop / mixMatch / mixSelf": f"{100 * mix['pop']:.1f} / {100 * mix['match']:.1f} / {100 * mix['self']:.1f}",
    "sdPopIdx / sdMatchIdx": f"{VAR.loc['popularity', 'within_sd_median']:.2f} / {VAR.loc['demand match', 'within_sd_median']:.2f}",
    "Section 6: growCumTop / growNoMomTop / growSFTop / growRealTop": " / ".join(f"{sr.loc[v, 'top10_mean']:.3f}" for v in
        ["cumulative only (refit)", "no momentum (refit)", "SF full (refit)", "REAL"]),
    "Section 7: platMatchTwoRet / platMatchFourRet / platPopHalfRet": " / ".join(f"{pf.loc[v, 'd_retention_pp']:.2f}" for v in ["match x2", "match x4", "pop x0.5"]),
}
show(pd.DataFrame([dict(new_macro=k, value=v) for k, v in new.items()]))
chk = [("fiveShPop", 100 * T1.loc["popularity", "in-sample"]), ("fiveShFit", 100 * coarse.loc["fit (quality + match)", "share"]),
       ("fiveShSelf", 100 * T1.loc["self-supply", "in-sample"]), ("fiveShHabit", 100 * T1.loc["habit", "in-sample"]),
       ("typeLightPop", 100 * R.loc["P5_indiv_light", "popularity"]), ("typeMajorSelf", 100 * R.loc["P5_org_major", "self-supply"]),
       ("sampDecHabit", 100 * R.loc["D5", "habit"]), ("sampAuthWFit", 100 * R.loc["P5a", "fit (quality + match)"])]
show(pd.DataFrame([dict(macro=k, in_paper=macros.get(k), recomputed=f"{v:.1f}") for k, v in chk]))
""")

md(r"""
### 10.4 `5_results.tex` 需要修改的地方
1. **全节改为五组口径。** "inertia" 改为自供给加习惯，宏换成 `\fiveSh*`。
2. **5.1：** 补充五组 LR 和潜类别的一句话。
3. **5.2：** 由现有的 "How large could fit be?" 改写，结尾加副本结果，作为过渡。
4. **新增 5.3：** 流行组内拆分，配图 2。
5. **5.4：** 补充集合内变异、选择分类（7.5% / 51%）、集约边际和 donor 安慰剂。
6. **新增 5.5（作者类型）和 5.6（样本选择）。**
7. **`sec:robustq` 移入附录 D。**
8. **Section 6：** 增长消融等 §7.1 的建议决定后再写；删去 todo 中"over-concentration"的旧解释，或者用 §7.1 的推测替换。
9. **Section 7：** 加入平台反事实和留存的段落。
""")

# =====================================================================================================================
md(r"""
## 11. 审稿人可能的追问

| 追问 | 回答与证据 |
|---|---|
| 这不就是马太效应 / 优先连接吗？ | 不完全是。累积采纳只占约四分之一，并且是强烈次线性的（$d^{0.34}$）；使用门槛、近期动量、可见度合计占一半以上（5.3）。只靠累积项生成的网络，头部集中只有真实的三分之一（Section 6）。 |
| 动量和累积哪个更重要？ | 份额相当，样本内和留出集上的排名互换，所以分解本身分不出来。区分来自质量固定的副本（只有动量起作用，5.2）和增长消融（去掉动量后集中减半，Section 6）。 |
| 流行份额高，是不是因为流行特征多、方差大？ | 份额反映的是预测能力。恢复检验（Section 4.6）中，选择只由质量和匹配生成时，分解报告 94% 的适合。§4.2 也显示流行指数的集合内变异更大，这正是它解释力更强的原因。 |
| 开发者是不是只选最热门的那个？ | 不是。被选者是集合中最流行者的比例只有约 7.5%（5.4）。 |
| 流行是不是质量的代理？ | 三层证据（5.2）：FE 上界下流行仍占一半；外部质量只解释 FE 的 7%；同权重副本中动量胜出。 |
| 流行是不是只决定能否进入考虑范围，之后由适合决定？ | 不是。只看已被使用过的候选（d≥1、d≥10），流行仍占 94%（5.4）。 |
| 匹配是不是共同潮流？ | donor 安慰剂（5.4）：donor 进入后，自己的匹配仍保留 4.5%。 |
| 结果是不是被少数高产账号驱动？ | 恰好相反。作者等权后适合更小（1.8%），轻度个人的流行份额最高（97%）（5.5、5.6）。 |
| 为什么只用首次采纳？ | 首次采纳对应"怎么挑"。加入重复选择后，习惯占 22–45%，适合仍然更小（5.6）。 |
| 早期采纳者是不是不同？ | 证据方向不一致，未在论文中报告（§9）。 |
| benchmark 只覆盖 118 个枢纽，代表性如何？ | 需要在正文中坦白。可以比较被覆盖与未覆盖枢纽的 FE 分布（待做）。 |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "/home/caroyf/myproject/ACMEC/notebooks/note4_results.ipynb")
print("cells:", len(cells))
