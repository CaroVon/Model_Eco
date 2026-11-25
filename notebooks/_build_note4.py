"""Builds notebooks/note4_results.ipynb (unexecuted). Executed afterwards by nbconvert in a Slurm job."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================================================================
md(r"""
# 笔记 4：What Drives Base-Model Choice（Section 5）的写作底稿

这份笔记把 **Section 5** 需要的结果集中在一处：每个结论用哪张表、哪个数字支撑，怎么解释，放正文还是附录。

- **口径：** v2.1 主规格 P5，即首次采纳、(d+1) PPS 候选集、五组特征（流行、通用质量、需求匹配、自供给、习惯）、按选择加权。区间是 50 次按作者的乘数 bootstrap。
- **与现有草稿的关系：** `ec_adoption/sections/5_results.tex` 写于 v2.1 之前，是四组口径，也没有作者类型和样本选择的结果。§10 列出需要改的地方。
- **代码：** 单元格已经运行过。重跑：`sbatch slurm/nb.sbatch notebooks/note4_results.ipynb`，大约需要 40 GB 内存；§3 读取 3,500 万行候选。
- 所有数字直接读自 `results/tables/v2/` 和 `processed/choices/v2_pps/v2_samples/`。

**目录**
0. 写作思路：要证明什么、按什么顺序、重点结论
1. 主结果：流行解释了几乎所有能解释的部分
2. 系数：开发者看重什么（外延边际、动量、匹配）
3. 为什么适合被看重却解释得少："匹配缩小范围，流行决定赢家"
4. 流行是不是隐藏的质量？（FE 上界、外部质量、同权重副本）
5. 适合是不是开发者特有的需求？（donor 安慰剂）
6. 谁在跟随流行？（作者类型）
7. 稳健性：样本选择、候选集设计、需求信号、文本、旧卡片
8. （可选）匹配好的选择是否更持久？留存
9. 正文 / 附录的分配，以及表和图的清单
10. 草稿需要修改的地方与数字核对
11. 审稿人可能的追问
""")

md(r"""
## 0. 写作思路

### 0.1 这一节要证明什么
Section 4 给了方法，Section 5 要回答标题的问题："什么驱动了基座模型的选择？"读者读完应当接受四个层层递进的判断：
1. **流行解释了几乎所有能解释的部分**（§1）。这是主结论，必须给出区间、留出集和似然比检验，并且用另一种单位（潜类别，"多少比例的选择"）复核。
2. **这不是因为开发者不在乎适合，而是因为适合很少能把候选区分开**（§2、§3）。这一点决定了结论的含义：不是"开发者盲目跟风"，而是"在适合的候选中，靠流行决定选谁"。
3. **流行不是隐藏的质量，适合也不是共同的潮流**（§4、§5）。这是排除两种替代解释。
4. **这个结论在典型开发者身上更强，并且对样本选择方向稳健**（§6、§7）。这是 v2.1 新增的部分，也回应了"少数作者"的质疑。

### 0.2 建议的小节结构

| # | 小节标题（建议） | 一句话主张 | 关键数字 | 本笔记 |
|---|---|---|---|---|
| 5.1 | Popularity accounts for almost all of what can be explained | 流行 92%，适合 5%，自供给 3%，习惯 0.3%，质量约 0；留出集相同 | 份额与区间、LR、留出集增益、潜类别 | §1 |
| 5.2 | Fit is valued but rarely decisive | 每单位的匹配价值很高（任务匹配相当于 degree ×246），但约 8 个候选同时匹配最好；流行指数在集合内的变异大得多 | MRS、并列数、集合内变异、选择分类 | §2、§3 |
| 5.3 | Is popularity hidden quality? | 即使把枢纽 FE 全部算作质量，流行仍占一半；外部质量只解释 FE 的 7%；同权重副本中动量胜出 | 上界三档、$q$、副本系数 | §4 |
| 5.4 | Is fit developer-specific? | 加入他人的需求后，自己的匹配份额基本保留 | 4.5% 对 2.4% | §5 |
| 5.5 | Who follows popularity? | 典型开发者（轻度个人，占一半）流行 97%；适合份额随账号规模上升；自供给是组织行为 | 作者类型表、LR 检验 | §6 |
| 5.6 | Robustness | 方向在所有口径下一致；主结果的适合份额是各口径中的偏高端；事先确定的规则如实报告 | 样本选择表；候选集设计 | §7 |
| （5.7） | Do well-matched choices last? | 对个人作者，匹配好的父模型更常被继续使用 | 作者 FE 留存系数、福利 | §8 |

5.7 也可以放到 Section 6 开头，作为"后果"的第一部分；或者放到附录 E，正文只用一段话。§9 给出建议。

### 0.3 重点结论（按重要性排序）
1. **主结论：** 流行占似然增益的 91.6%（95% 区间 91.1–91.9），适合 5.2%，自供给 2.9%，习惯 0.3%，通用质量接近 0；留出集上几乎相同。
2. **机制：** 适合被高度看重但很少起决定作用。"匹配缩小范围，流行决定赢家。"
3. **动量而非累积：** 同权重副本中，起作用的是近 30 天的采纳（系数 0.84），而不是总采纳数（−0.08）。这对 EC 读者很重要：它区分了"动量 / 羊群"与"从总数中学习质量"。
4. **质量的上界：** 即使把全部枢纽 FE 算作质量，流行仍占 51%；benchmark 能解释的只有其中 7%。
5. **异质性（新）：** 典型开发者几乎完全跟随流行；适合份额随账号的规模和专业程度上升。
6. **稳健性（新）：** 所有替代口径都让适合变小或让习惯挤占流行，没有一个让适合变大。

### 0.4 写作上的几个原则
- **每个结论都给出"它不是什么"。** 例如：流行主导 ≠ 开发者不在乎适合（5.2）；流行 ≠ 质量（5.3）；匹配 ≠ 共同潮流（5.4）。EC 读者最在意替代解释是否被排除。
- **份额的单位要始终一致：** "似然增益的份额"。第一次出现时翻译一次："知道流行信号后，预测真实父模型的能力提高了多少"。用潜类别的"选择份额"作为补充单位。
- **不要用"盲目跟风"之类的词。** 结论是描述性的，"跟随流行"在质量难以观察时可以是理性的（Section 7 讨论）。
- **数字全部用宏**，`paper_tables.py` 已经生成了大部分，§10 列出还需要新增的宏。
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
print("main config:", P5["cfg"]); print(P5["info"])
""")

# =====================================================================================================================
md(r"""
## 1. 主结果：流行解释了几乎所有能解释的部分

### 1.1 五组 Shapley 份额（样本内与留出集）
下面的表和图是 5.1 的核心，即论文中的 Table `tab_shapley`（需要更新为五组）。
- **样本内：** 份额与 50 次 bootstrap 的 95% 区间。
- **留出集：** 用训练集的估计预测最后 10% 的选择。一组特征只有在能预测**未来**选择时才能在这里得分。
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
ax.set_title("What explains base-model choice (first adoptions, five groups)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.2 每一组都显著：似然比检验
从 32 次子集拟合可以直接得到"去掉一组"的似然比统计量：
$$LR_g = 2\,[\ell(\hat\theta_{\mathcal G}) - \ell(\hat\theta_{\mathcal G\setminus g})] = 2\,[G(\mathcal G) - G(\mathcal G\setminus g)],$$
自由度是该组的特征数。

这一点在正文里的作用是**把"显著"与"重要"分开**：每一组都显著（样本很大），所以问题不是"开发者是否回应适合"，而是"适合能解释多少选择"。这正是 Shapley 份额回答的问题。
""")

code(r"""
fits, groups = P5["fits"], P5["groups"]
G = lambda S: fits[frozenset(S)]["ll"] - fits[frozenset()]["ll"]
allg = list(groups)
from scipy.stats import chi2
lr = []
for g in allg:
    st = 2 * (G(allg) - G([h for h in allg if h != g]))
    df = len(groups[g])
    lr.append(dict(group=LAB[g], features=df, LR=st, p=chi2.sf(st, df) if df else np.nan))
show(pd.DataFrame(lr).round(2))
info = P5["info"]; h = P5["nofe"]["hold"]
print(f"holdout: gain {h['mean_ll'] - P5['u0h']:.3f} nats/choice (x{np.exp(h['mean_ll'] - P5['u0h']):,.0f}), "
      f"top-1 {pct(h['top1'])}, top-10 {pct(h['top10'])}; pseudo-R2 in-sample {1 - P5['nofe']['ll'] / P5['u0']:.3f}")
""")

md(r"""
### 1.3 换一个单位：多少比例的选择是流行驱动的（潜类别模型）
Shapley 份额的单位是"似然增益"。审稿人可能会问："这是否等于多少开发者在跟风？"

潜类别模型给出另一个单位。假设每次选择来自几种"模式"之一（只看流行、只看质量、只看匹配、均匀、自供给），用 EM 估计每种模式的比例，再算每种模式负责的**选择份额**。两个单位给出同样的排序，结论就不依赖单位的选择。下面还附上半合成检验：在已知比例的模拟数据上，方法能否恢复比例。
""")

code(r"""
mx = tab("v2_mixture_planA").set_index("quantity").value
mix = pd.Series({k.replace("share_", ""): float(mx[k]) for k in mx.index if k.startswith("share_")})
show((mix * 100).round(2).rename("share of choices (%)").to_frame())
ms = tab("v2_mixsim_planA")
show(ms.groupby(["dgp", "mode"])[["realised_share", "est_share", "abs_error"]].mean().round(4))
""")

md(r"""
**写作要点（5.1）：**
- **第一段直接给主结论**，配 `tab_shapley`（改为五组）或上面的图：流行 91.6%（91.1–91.9），适合 5.2%，自供给 2.9%，习惯 0.3%，质量 < 0.01%。
- **紧接着给留出集。** 份额几乎不变，说明排序不是样本内拟合的产物。
- **用一句话把"显著"和"重要"分开**（LR 检验），引出 5.2。
- **潜类别结果放一句话即可：** "按选择计，92% 的选择归于流行模式"。细节放附录。
- **自供给要在这里第一次解释：** 它是"选自己的模型"，主要是组织行为（§6）。旧草稿中的"惯性 3.2%"几乎全是自供给。
""")

# =====================================================================================================================
md(r"""
## 2. 系数：开发者看重什么

### 2.1 主模型系数与边际替代率
系数是效用权重，单位各不相同。为了可比，换算成"等效的 degree 倍数"：某个特征变化一次（二值特征从 0 到 1；连续特征增加 1 个标准差）带来的效用变化，等于把 $(1+d)$ 乘以多少倍。即 $\exp(\Delta x\cdot\theta_f/\theta_{\log d})$，参考基准是 `log_deg`。
""")

code(r"""
cf = tab("v2s_P5_clogit")
mrs = tab("v2_mrs")
m1 = mrs[mrs.ref == "log_deg"].set_index("feature")[["dx", "multiplier", "mult_lo", "mult_hi"]]
T2 = cf.set_index("param")[["group", "coef", "se_cl", "lo", "hi"]].join(m1)
show(T2.round(3))
""")

md(r"""
### 2.2 三个值得写进正文的读法
1. **外延边际极大：** `has_deg` 的系数 4.64，即从"没人用过"到"有人用过"，概率乘以约 100 倍；换算成 degree 倍数是几十万倍。已被使用过，是最强的单一信号。
2. **动量比累积更重要：** 近 30 天的采纳（`lrec30`）增加 1 个标准差，相当于 degree 乘以约 8.5 倍；而 `log_deg` 本身是次线性的（$P\propto d^{0.34}$）。§4.3 的同权重副本给出更干净的证据。
3. **匹配每单位的价值很高：** 候选的任务出现在作者历史中（`task_hist`），相当于 degree 乘以约 246 倍（区间 149–407）。这说明适合份额小**不是**因为开发者不在乎匹配，引出 §3。
""")

code(r"""
for f in ["has_deg", "lrec30", "major_org", "task_hist", "lang_hist", "txt_sim", "same_author"]:
    if f in m1.index:
        r = m1.loc[f]
        print(f"{f:12s} multiplier on (1+d): {r.multiplier:,.1f}  [{r.mult_lo:,.1f}, {r.mult_hi:,.1f}]  (dx = {r.dx:.3g})")
""")

md(r"""
**写作要点：** 系数表放附录 D（`tab_coef`）。正文只用上面三个读法，每个一句话，并用宏引用边际替代率（`\mrsTask`、`\mrsRec`、`\mrsOrg` 已存在）。
""")

# =====================================================================================================================
md(r"""
## 3. 为什么适合被看重却解释得少

### 3.1 问题
§2 显示匹配每单位的价值很高，但 §1 显示它只解释 5% 的增益。两者并不矛盾：一组特征能解释多少选择，取决于它的**效用指数在候选集内变异多大**（Section 4.4 的第 (iii) 点）。如果大多数候选的匹配得分相同，匹配就无法区分它们，不论系数多大。

### 3.2 证据一：各组指数在候选集内的变异
对训练集的每次选择，计算每组的效用指数 $I_{g,kl}=\hat\theta_g^\top x_{kl,g}$ 在候选集内的标准差和极差。这是 Section 5.2 最直接的证据，旧草稿只用文字描述了这一点（"degree 跨越五个数量级"）。
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
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, g in enumerate(["pop", "match", "self", "qual"]):
    v = I[g]
    mu = np.add.reduceat(v, st) / ln
    sd = np.sqrt(np.maximum(np.add.reduceat(v * v, st) / ln - mu ** 2, 0))
    ax.hist(np.clip(sd, 0, 4), bins=80, histtype="step", lw=1.6, color=C5[i], label=LAB[g], density=True)
ax.set_xlabel("within-set standard deviation of the group's utility index (one value per choice; clipped at 4)")
ax.set_ylabel("density"); ax.set_yscale("log")
ax.set_title("Popularity varies across candidates far more than fit does", loc="left")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
### 3.3 证据二：多少候选同时"匹配最好"，以及被选者是哪一类
在每次选择的候选中（只看已被使用过的候选，即 $d>0$），判断被选的父模型是否是匹配指数最高的（"适合型"）、流行指数最高的（"流行型"）、两者都是，或者都不是。并列时随机打破，报告期望份额。
""")

code(r"""
cl = tab("v2_classify")
show(cl.round(4))
""")

md(r"""
**怎么读：**
- 有需求信号的选择中，**平均约 8 个候选并列匹配最好**。匹配能把候选集从约 300 个缩小到一个小的子集，但无法在子集中选出赢家。
- 被选者恰好是集合中流行指数最高者的选择只占约 7.5%，"都不是"占 51%。所以"跟随流行"是概率意义上的：流行提高被选的机会，而不是每次都选最热门的那个。正文里应当这样措辞，避免读者把结论理解为"人人选最热门"。
- 这两点合起来就是 5.2 的核心句："**匹配缩小范围，流行决定赢家**"（fit narrows the field; popularity picks the winner）。

**写作要点（5.2）：**
- 先给边际替代率（匹配的价值很高），再给集合内变异和并列数（匹配很少能区分候选），最后给核心句。
- §3.2 的变异表是新的证据，建议在正文中用一句话引用中位数，表放附录。需要新增宏（§10）。
""")

# =====================================================================================================================
md(r"""
## 4. 流行是不是隐藏的质量？

benchmark 分数只覆盖很少的候选，所以"质量份额约 0"可能只是没测到。三种证据，强度依次递增：

### 4.1 上界：把枢纽 FE 全部算作质量
加入 500 个枢纽的固定效应（Section 4.5 定义 4），作为 Shapley 的额外玩家。FE 吸收枢纽所有不变的属性，包括不可观测的质量，也包括品牌、教程、默认地位。把它全部算作质量，是对质量**最宽松**的解读。
""")

code(r"""
show(tab("v2_bounds").round(4))
fit_ = tab("v2_pps_clogit_fit").set_index("model")
print(f"holdout gain: no FE {fit_.loc['full', 'hold_gain']:.3f}, with FE {fit_.loc['full + FE', 'hold_gain']:.3f} nats/choice")
fe5 = sh[(sh.variant == "FE as a player") & (sh.partition == "pop / fit / ...")].pivot(index="group", columns="measure", values="share")
show((fe5 * 100).round(2))
""")

md(r"""
### 4.2 外部质量能解释 FE 的多少
用枢纽的外部质量（Open LLM Leaderboard v2 平均分、Arena 评分）回归 $\hat\phi_h$，控制参数量和再分发账号，并做噪声校正（Section 4.5 引理 4）。
""")

code(r"""
qe = tab("v2_quality_ext")
show(qe[[c for c in ["source", "controls", "hubs", "share_of_fe", "r2_noise_corrected", "quality_part_share"] if c in qe]].round(3))
""")

md(r"""
### 4.3 质量固定时：同权重副本
有 638 次首次采纳，候选集中同时有两个**同一份权重**的副本（权重哈希相同，由不同账号发布），并且其中一个被选中。质量在构造上完全相同。用两个副本的特征差做条件 logit，看是什么让一个副本胜出。
""")

code(r"""
show(tab("v2_identical_pairs").round(4))
""")

md(r"""
**怎么读：**
- **上界：** 即使把全部 FE 算作质量，流行仍占约 51%（样本内），适合（含质量）约 46%；留出集上 FE 的份额更小，适合为 36%。
- **FE 几乎不改善样本外预测：** 留出集增益从 6.625 增加到 6.632 nats。FE 在样本内"接管"了与之共线的流行特征的功劳，但它预测未来选择的能力并不比流行信号更强。
- **外部质量只能解释 FE 方差的约 7%**（118 个被覆盖的枢纽，噪声校正后）。按此拆分，适合份额只升到 7.0%（样本内）/ 6.3%（留出集）。
- **同权重副本：** 胜出的是近 30 天采纳更多的那个副本（系数 0.84，标准误 0.08），而不是总采纳更多的那个（−0.08，标准误 0.10）；被选副本的总 degree 更高的比例只有 48.6%，相当于抛硬币。**质量固定时，开发者跟随的是当前的动量。**

**写作要点（5.3）：**
- 三层证据按"上界 → 外部质量 → 质量固定"的顺序写，最后以副本结果收尾，它是最干净的证据。
- **待办：** benchmark 覆盖率只有 118/500 个枢纽，需要讨论代表性（草稿中已有 todo）。
- 副本的结论与 Section 7 的"社会学习还是羊群"直接相关：从总数中学习质量，应当看总采纳；看近期采纳，更像动量或羊群。
""")

# =====================================================================================================================
md(r"""
## 5. 适合是不是开发者特有的需求？

**替代解释：** 匹配份额可能只是共同的潮流。如果某个月大家都在微调指令跟随的文本模型，任何开发者的历史都会"匹配"那个月流行的模型。

**检验（donor 安慰剂）：** 把每个开发者的需求换成同月同模态另一位作者的需求，重算匹配特征。
- 单独使用 donor 的需求时，它得到多少份额？这衡量共同潮流有多大。
- 自己的需求和 donor 的需求同时进入模型时，自己的匹配还剩多少？这衡量开发者特有的需求。
""")

code(r"""
show(tab("v2_placebo").round(4))
show(tab("v2_placebo_joint").round(4))
""")

md(r"""
**怎么读：**
- 单独使用 donor 需求，匹配份额为 3.1%，说明共同潮流确实存在。
- 两者同时进入时，自己的匹配保留 4.5%，donor 只有 2.4%，自己的匹配系数几乎不变。
- donor 的需求挤占的是**流行**的份额（92% → 90%），而不是自己的匹配。
- **约 4.5 个百分点的增益来自开发者特有的需求。**

**写作要点（5.4）：** 一段即可，宏已存在（`\placeboAlone`、`\jointOwn`、`\jointDonor`）。这是对"匹配只是潮流"的直接回应。
""")

# =====================================================================================================================
md(r"""
## 6. 谁在跟随流行？（v2.1 新增）

### 6.1 按作者类型分开估计
六类作者分别估计完整模型和 32 次子集拟合，各做 50 次 bootstrap。类型由 HF API 的账号类型、大机构名单、再分发规则和发布量确定。跨类型比较用份额，不用系数，因为 logit 的尺度在类型之间可以不同。
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
    ax.text(101, i, f"n = {int(n):,} ({100 * s:.0f}%)", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(TT.index); ax.invert_yaxis(); ax.set_xlim(0, 125)
ax.set_xlabel("share of the log-likelihood gain (%), first adoptions")
ax.set_title("Who follows popularity: shares by author type", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.42, -0.42))
plt.tight_layout(); plt.show()
""")

md(r"""
**怎么读：**
- **典型开发者几乎完全跟随流行：** 轻度个人（少于 10 个模型）占首次采纳的一半，流行占 97.1%，适合只有 2.1%。
- **适合份额随账号的规模和专业程度上升：** 中等个人 7.8%，高产个人 11.2%，其他组织 10.4%，再分发账号 14.8%，大机构 17.3%。
- **自供给是组织行为：** 大机构的自供给份额达 21.5%，轻度个人只有 0.8%。
- **合并估计被显著拒绝：** $LR=38{,}158$，自由度 130。
- **与主结果的关系：** 主结果中 5.2% 的适合份额，主要来自少数高产账号和组织。按作者等权，适合份额降到 1.8%（§7）。

**写作要点（5.5）：**
- 这是本轮最值得进正文的新发现，建议配一张图（上面的堆叠条形图），或者用 `tab_samples` 的"作者类型"部分。
- **措辞：** "fit matters more for experienced and organizational developers"，避免写成"个人开发者盲目"。可以连到 Section 7：信息成本低（或评估能力强）的开发者更多地依据适合来选择。
- **大机构只有 254 次选择，** 区间较宽（流行 52.5–67.9），提到时注明样本量。
- **"再分发账号"是按行为定义的**（以上传副本为主），并不都是组织账号，需要在 Section 3 或附录 A 说明。
""")

# =====================================================================================================================
md(r"""
## 7. 稳健性

### 7.1 样本选择与"少数作者"（v2.1）
**三个问题：** 加入重复选择后会怎样？少数高产作者是否驱动了结果？去掉头部作者会怎样？判断规则是在运行前确定的（Section 4.7）：流行份额与主结果相差不超过 10 个百分点，并且适合/流行比值在主结果的 1/3 到 3 倍之间。
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
show(tab("v2s_ladder").round(3))
""")

code(r"""
fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
idx = list(RS.index)
yy = np.arange(len(idx))
for ax, (col, lab_, colr) in zip(axs, [("popularity", "popularity share (%)", C[0]), ("fit (quality + match)", "fit share (%)", C[1])]):
    v = RS[col] * 100
    ax.scatter(v, yy, color=colr, s=28, zorder=3)
    if col == "popularity":
        ax.errorbar(v, yy, xerr=[(RS[col] - RS[col + " lo"]) * 100, (RS[col + " hi"] - RS[col]) * 100], fmt="none",
                    ecolor="black", lw=1, capsize=2)
    ax.axvline(v.iloc[0], color=GRAY, ls="--", lw=1)
    ax.set_xlabel(lab_)
axs[0].set_yticks(yy); axs[0].set_yticklabels(idx); axs[0].invert_yaxis()
axs[0].set_title("Sample selection: every alternative makes fit smaller or habit larger", loc="left")
plt.tight_layout(); plt.show()
""")

md(r"""
**怎么读：**
- **按规则，只有"去掉前 1% 作者"通过。** 但所有偏离的方向一致：
  - 加入重复选择后，习惯挤占流行（全部决策中习惯 21.9%，全部上传中 45.4%）；
  - 作者等权或去掉头部后，适合进一步变小（1.8%、3.5%）。
- **没有任何口径让适合变得比主结果更重要。** 正文应写："流行主导在所有口径下都成立；主结果的适合份额是各口径中的偏高端。"不要笼统地写"结果稳健"。
- **扣除自身 degree 的修正**改变份额不到 1 个百分点（D5 与 D5raw）。
- U5 的 bootstrap 次数见上表 `reps` 列；如果不足 50，作业完成后重跑 collect 即可更新。

### 7.2 候选集设计
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
PPS 与分层设计一致；均匀设计的适合份额更高，原因是均匀候选集几乎不含热门竞争者，选择变成了"已知 vs 未知"的比较（笔记 3 §2.8）。草稿中的 `sec:robustq` 小节已经写好，只需改为五组口径。

### 7.3 需求信号、文本门槛、旧卡片、10% 样本
""")

code(r"""
sg = tab("v2_signal_samples"); show(sg[sg.partition == "pop / fit / inertia"].round(4))
show(tab("v2_text_threshold").round(4))
show(tab("v2_old_cards").round(4))
v10 = tab("v2_validate_10pct")
print(f"10% vs full: {int(v10.in_10pct_ci.sum())} of {len(v10)} coefficients inside the 10% CI; median SE ratio {v10.se_ratio.median():.2f}")
""")

md(r"""
**怎么读：**
- **需求信号：** 只有 67% 的首次采纳带有需求信号。在这些选择上，匹配份额是 8.3%；按作者经验和机构重新加权后是 6.4%。主结果在这个意义上是适合份额的下界：没有留下痕迹的需求不会被归给任何组。
- **文本门槛：** 最低词数从 5 提高到 50，文本相似度的系数变大（1.7 → 16.5），但匹配份额不变（5.2%–5.3%）。
- **旧卡片：** 39 个枢纽在被大量采用之前的卡片版本，与当前版本的相关系数是 0.51；用旧卡片时，被选父模型与其他候选的相似度差距方向相同。
- **全量与 10% 样本：** 系数基本一致。

**写作要点（5.6）：**
- 正文用两段：样本选择一段（配 `tab_samples`），候选集设计一段（现有的 `sec:robustq`）。
- 需求信号、文本门槛、旧卡片、10% 样本放附录 D，正文一句话带过。
""")

# =====================================================================================================================
md(r"""
## 8. （可选）匹配好的选择是否更持久？留存

这部分回答"挑得好不好"，可以作为 5.7 的一小节，或者放到 Section 6 开头、附录 E。方法见笔记 3 §10：作者固定效应的条件 logit，因变量是下一次决策是否仍用同一父模型。
""")

code(r"""
rb = tab("v2s_retention_bytype")
fe_ = rb[(rb.model == "conditional logit, author FE") & rb["var"].isin(["I_pop", "I_match", "I_self"])]
show(fe_.pivot(index="stratum", columns="var", values="coef").round(3))
show(fe_.pivot(index="stratum", columns="var", values="z").round(1))
show(tab("v2s_retention_bytype_outcomes").round(4))
show(tab("v2s_retention_welfare").round(4))
""")

md(r"""
**怎么读：**
- **对个人作者，** 在同一作者内部比较，匹配好的父模型更常被继续使用（轻度个人 +0.47，中等个人 +0.11）。
- **对高产账号和组织，** 匹配不起作用，它们的留下或换掉反映的是生产流程。
- **留存福利：** 个人作者如果改选匹配最好的候选，每次不匹配的选择留存概率提高 2.4 个百分点，相当于基线的 8.6%。

**建议：** 正文放一段，作为"跟随流行是否有代价"的初步证据，自然地过渡到 Section 6。表放附录 E，并删去旧草稿中"匹配系数为负"的表述。
""")

# =====================================================================================================================
md(r"""
## 9. 正文 / 附录的分配，以及表和图

| 内容 | 位置 | 表 / 图 | 状态 |
|---|---|---|---|
| 五组 Shapley 份额（样本内、留出集、区间） | 正文 5.1 | Table `tab_shapley`（需改为五组）或 §1 的图 | 需要更新 `paper_tables.py::tab_shapley` |
| LR 检验、留出集指标 | 正文 5.1（文字） | — | 需要新增宏（五组 LR） |
| 潜类别（选择份额） | 正文 5.1 一句；附录 | — | 已有结果，需要新增宏 |
| 系数与边际替代率 | 附录 D | `tab_coef` | 已有 |
| 集合内变异、并列数、选择分类 | 正文 5.2（文字）；附录 | §3 的表或图 | 需要新增宏 |
| 质量上界、外部质量、副本 | 正文 5.3 | `tab_bounds` | 已有 |
| donor 安慰剂 | 正文 5.4（文字） | — | 已有宏 |
| 作者类型 | 正文 5.5 | §6 的堆叠图，或 `tab_samples` 的下半部分 | 表已有；图需要导出 |
| 样本选择 | 正文 5.6 | `tab_samples` 的上半部分 | 已有 |
| 候选集设计 | 正文 5.6 | `tab_q`（需改为五组或保持四组并注明） | 已有 |
| 需求信号、文本、旧卡片、10% | 附录 D | — | 已有 |
| 留存 | 5.7 一段或 Section 6；附录 E | — | 已有宏（`ret*`） |

**图的建议：** 正文放两张图。
- Figure A：§1 的份额条形图（样本内 vs 留出集）。它比表更直观，可以替代 `tab_shapley`，表移到附录。
- Figure B：§6 的作者类型堆叠图。

两张图都可以从本笔记导出为 PDF，放到 `ec_adoption/figures/`。
""")

# =====================================================================================================================
md(r"""
## 10. 草稿需要修改的地方与数字核对

**`5_results.tex` 需要修改：**
1. **全节改为五组口径：** "inertia" 改为自供给 + 习惯；`\shInertia` 等宏改为 `\fiveShSelf`、`\fiveShHabit`（已生成）。
2. **5.1 补充：** 五组的 LR（下表）、潜类别的一句话。
3. **5.2 补充：** 集合内变异的证据（§3.2）、选择分类（7.5% / 51%）。
4. **新增 5.5"谁在跟随流行"、5.6 的样本选择部分**（`tab_samples`），以及（可选）5.7 留存。
5. **`sec:robustq` 改为五组或注明四组；** 均匀设计的数字核对。
6. **删去 Section 5 中对"惯性"的旧解释。**

下面把本笔记依赖的已有宏与重算值对照，并列出建议新增的宏。
""")

code(r"""
chk = [("fiveShPop", 100 * T1.loc["popularity", "in-sample"]), ("fiveShFit", 100 * coarse.loc["fit (quality + match)", "share"]),
       ("fiveShSelf", 100 * T1.loc["self-supply", "in-sample"]), ("fiveShHabit", 100 * T1.loc["habit", "in-sample"]),
       ("typeLightPop", 100 * R.loc["P5_indiv_light", "popularity"]), ("typeMajorSelf", 100 * R.loc["P5_org_major", "self-supply"]),
       ("sampDecHabit", 100 * R.loc["D5", "habit"]), ("sampAuthWFit", 100 * R.loc["P5a", "fit (quality + match)"])]
show(pd.DataFrame([dict(macro=k, in_paper=macros.get(k), recomputed=f"{v:.1f}") for k, v in chk]))
new = {
    "lrSelf / lrHabit / lrMatch5 / lrPop5": "five-group LR statistics (§1.2)",
    "mixPop / mixMatch / mixSelf": f"latent-class choice shares: {100 * mix['pop']:.1f} / {100 * mix['match']:.1f} / {100 * mix['self']:.1f}",
    "sdPopIdx / sdMatchIdx": f"median within-set SD of the index: {VAR.loc['popularity', 'within_sd_median']:.2f} / {VAR.loc['demand match', 'within_sd_median']:.2f}",
    "clsPopType / clsNeither": "chosen is the most popular / neither (classification, §3.3)",
}
show(pd.DataFrame([dict(new_macro=k, content=v) for k, v in new.items()]))
""")

# =====================================================================================================================
md(r"""
## 11. 审稿人可能的追问

| 追问 | 回答与证据 |
|---|---|
| 流行份额高，是不是因为流行特征多、方差大？ | 份额反映的是预测能力，不是特征个数。恢复检验（Section 4.6）：选择只由质量和匹配生成时，分解报告 94% 的适合，并没有制造流行份额。§3 也直接显示流行指数的集合内变异更大，这正是它解释力更强的原因。 |
| 开发者是不是只是选了最热门的那个？ | 不是。被选者是集合中最流行者的比例只有约 7.5%（§3.3）。流行提高被选的概率，而不是决定性地选最热门者。 |
| 流行是不是质量的代理？ | 三层证据（§4）：FE 上界、外部质量只解释 7%、同权重副本中动量胜出。 |
| 匹配是不是共同潮流？ | donor 安慰剂（§5）：自己的匹配在 donor 进入后仍保留 4.5%。 |
| 结果是不是被少数高产账号驱动？ | 恰好相反：作者等权后适合份额更小（1.8%），轻度个人的流行份额最高（97%）（§6、§7）。 |
| 为什么只用首次采纳？ | 首次采纳对应"怎么挑"。加入重复选择后，习惯占据 22–45% 的增益，适合仍然更小（§7.1）。 |
| 均匀候选集为什么给出不同的权重？ | 均匀候选集几乎不含热门竞争者，衡量的是"已知 vs 未知"的边际，而不是开发者实际做的比较（§7.2）。 |
| benchmark 只覆盖 118 个枢纽，代表性如何？ | 需要在正文中坦白。可以比较被覆盖与未覆盖枢纽的 FE 分布（待做）。 |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "/home/caroyf/myproject/ACMEC/notebooks/note4_results.ipynb")
print("cells:", len(cells))
