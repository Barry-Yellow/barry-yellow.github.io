# 风险价格与风险溢价

**风险价格（risk prices/prices of risk）是一个笼统的概念，因为风险的度量有很多种，在不同的语境下风险价格代表不同的东西。**

> Whether "price of risk" refers to the coefficient of the covariance or to the coefficient of the beta must be determined by context. &mdash;Back (2017, p.139)

**风险溢价（risk premia）指的是某个资产在剔除无风险收益后的收益率，即有风险的那部分，常常被表示为 $\E(R) - R_f$ 的形式。**

风险价格既可以出现在 SDF 的框架中，也可以出现在 β 表达式的框架中；而风险溢价则只在 β 表达式的框架中出现。**当两个框架同时出现时，通常因子的风险价格指 SDF 载荷，因子的风险溢价则指 β 表达式中 β 的系数。**

下面对这两个框架做一些统一的设定，并分别给出风险价格和风险溢价的例子。

## 设定

假设长度为 $T$ 的时间窗口内有 $K$ 个因子 $\bm{f} = (\bm{f}_{\! 1},\ \bm{f}_{\! 2},\ \cdots,\ \bm{f}_{\! T})^{\top}$，其中 $\bm{f}_{\! t} = (f_{1t},\ f_{2t},\ \cdots,\ f_{Kt})^{\top},\ t=1,\ 2,\ \cdots,\ T$。同时，考虑 $N$ 个 test assets（超额收益率）$\bm{R}^{e} = (\bm{R}_1^{e},\ \bm{R}_2^{e},\ \cdots,\ \bm{R}_{T}^{e})^{\top}$，其中 $\bm{R}_t^{e} = (R_{1t}^{e},\ R_{2t}^{e},\ \cdots,\ R_{Nt}^{e})^{\top}$。

### SDF

我们通常用因子来构造线性形式的 SDF $m_t$ 来满足 test assets 的定价条件：

$$
\begin{align}
&m_t = 1 - [\bm{f}_{\! t} - \E(\bm{f}_{\! t})]^{\top} \bm{b} \label{1} \\
&\text{s.t.}\quad \E(m_t \bm{R}_t^{e}) = \bm{0}_{N} \label{2}
\end{align}
$$

其中 $\bm{b}$ 为 **SDF 载荷（SDF loadings）**。

把 $\eqref{1}$ 式代入限制条件 $\eqref{2}$ 可以解得

$$
\begin{align}
\bm{\mu}_{\bm{R}}^{e} = \bm{C} \bm{b} \label{3}\\
\end{align}
$$

其中 $\bm{\mu}_{\bm{R}}^{e} := \E(\bm{R}_t^{e})$，$\bm{C}$ 为 $\bm{R}_t^{e}$ 与 $\bm{f}_{\! t}$ 之间的协方差矩阵。

从 $\eqref{3}$ 式我们可以看出，**如果用协方差作为风险的度量，$\bm{b}$ 即每单位风险对应的期望收益率，因此可以被称作因子的风险价格。**

> [!NOTE|label:注意]
> 以下 $\mu_{x}$ 均代表某一变量 $x$ 的期望，且期望默认消除变量的时间维度。

### β 表达式

在传统的因子模型中，test assets 的期望收益率可以写成

$$
\begin{equation}
\bm{\mu}_{\bm{R}}^{e} = \bm{\beta} \bm{\lambda} \label{4} \\
\end{equation}
$$

其中 $\bm{\beta}$ 为资产在因子上的**风险暴露（risk exposure）**，$\bm{\lambda}$ 为因子的风险溢价。

仅从 $\eqref{4}$ 式我们是看不出为什么 $\bm{\lambda}$ 是风险溢价的，但**如果用 β 作为风险的度量，$\bm{\lambda}$ 其实也是每单位风险对应的期望收益率，因此也可以被称作因子的风险价格。**

当 SDF 的框架和 β 表达式的框架一同出现时，我们需要注意区分 $\bm{b}$ 和 $\bm{\lambda}$ 的名字，比如 Kozak 等（2020）的做法是将前者称作风险价格，后者称作风险溢价。

> Another important difference between our approach and much of this recent machine learning literature in asset pricing lies in the objective. Most papers focus on estimating **risk premia**, i.e., the extent to which a stock characteristic is associated with variation in expected returns. In contrast, we focus on estimation of **risk prices**, i.e., the extent to which the factor associated with a characteristic helps price assets by contributing to variation in the SDF. &mdash;Kozak et al. (2020, p.4)

下面我将通过介绍风险暴露的定义以及表达式，推导出 $\bm{\lambda}$ 为什么可以被称作因子的风险溢价。

#### 风险暴露

**风险暴露 $\bm{\beta}$ 是将收益率回归到因子得到的回归系数**：

$$
\begin{equation}
\bm{R}_t^{e} = \bm{a} + \bm{\beta} \bm{f}_{\! t} + \bm{\epsilon}_t \label{5}\\
\end{equation}
$$

其中 $\bm{a}$ 为截距项，$\bm{\epsilon}_t$ 为噪声向量。

因此在 OLS 假设下，$\bm{\beta} = (\bm{f}^{\top} \bm{f})^{-1} \bm{f}^{\top} \bm{R}$。<strong>当我们假设 $\bm{\mu}_{\bm{f}} = \bm{0}_{K}$</strong>，风险暴露可以写成更为简洁的形式：

$$
\begin{equation}
\bm{\beta} = \bm{\Sigma}_{\bm{f}}^{-1} \bm{C} \label{6} \\
\end{equation}
$$

其中 $\bm{\Sigma}_{\bm{f}}$ 为因子之间的协方差矩阵。

#### 因子的风险溢价

先给出结论：**假设 $\bm{f}$ 中一个因子 $\bm{f}_{\! k}$ 是收益率（不是超额收益率），对应的 $\lambda_k$ 就是因子的风险溢价。**

对 $\eqref{5}$ 式取期望，我们有

$$
\begin{equation}
\bm{\mu}_{\bm{R}}^{e} = \bm{a} + \bm{\beta} \bm{\mu}_{\bm{f}} \label{7}\\
\end{equation}
$$

用 $\eqref{5}$ 式减去 $\eqref{7}$ 式，我们得到

$$
\begin{equation}
\bm{R}_t^{e} = \bm{\mu}_{\bm{R}}^{e} + \bm{\beta} (\bm{f}_{\! t} - \bm{\mu}_{\bm{f}}) + \bm{\epsilon}_t \label{8}\\
\end{equation}
$$

假设存在无风险收益率 $\bm{R}^{f} = (R_1^{f},\ R_2^{f},\ \cdots,\ R_{T}^{f})^{\top}$，$\bm{f}_{\! k} - \bm{R}^{f}$ 也是超额收益率，那么它同样应该被定价：

$$
\begin{equation}
f_{kt} - R_t^{f} = (\mu_k - R_t^{f}) + \bm{\beta}_k (\bm{f}_{\! t} - \bm{\mu}_{\bm{f}}) + \epsilon_t \label{9}\\
\end{equation}
$$

其中 $\mu_k:= \E(\bm{f}_{\! k})$，$\bm{\beta}_k$ 是 $\bm{f}_{\! k} - \bm{R}^{f}$ 在不同因子上的风险暴露，是一个 $1 \times K$ 的行向量。

$\eqref{9}$ 式成立的条件为：$\bm{f}_{\! k} - \bm{R}^{f}$ 在 $\bm{f}_{\! k}$ 上的风险暴露 $\beta_{kk} = 1$ 而其他风险暴露为 $0$，这意味着

$$
f_{kt} - R_t^{f} = \beta_{kk} (f_{kt} - R_t^{f}) + \epsilon_t \implies \mu_k - \mu_{R^{f}} = \beta_{kk} (\mu_k - \mu_{R^{f}})\\
$$

说明 $\bm{\beta}_{k}$ 在 $\eqref{4}$ 式中的系数 $\lambda_k = \mu_k - \mu_{R^{f}}$，即 $\bm{\lambda}$ 是因子的风险溢价。

以上推导都建立在**因子是收益率**的情况下，**当因子不是收益率时，$\bm{\lambda}$ 实际上并不是因子的风险溢价**，只是我们为了方便或者区分风险价格而统称它为风险溢价。

当因子是超额收益率时，我们可以很容易地证明 $\bm{\lambda}$ 就是因子的期望；而当因子不可交易时（比如 GDP），我们并不知道 $\bm{\lambda}$ 是什么。

> The price of risk corresponding to a **return factor** is the return's risk premium. The price of risk corresponding to an **excess return** is the mean of the excess return. In contrast, for **general factors**, the prices of risk are not determined by theory. &mdash;Back (2017, p.136)

## 区别与联系

> [!NOTE|label:注意]
> 以下我们将沿用大多数文献采用的方式，即用风险价格指代 $\bm{b}$，用风险溢价指代 $\bm{\lambda}$。

假设因子的期望均为 $0$，由 $\eqref{3}$、$\eqref{4}$、$\eqref{6}$ 式我们可以推出风险溢价与风险价格之间的关系：

$$
\begin{equation}
\bm{\lambda} = \bm{\Sigma}_{\bm{f}} \bm{b} \label{10} \\
\end{equation}
$$

当因子之间是正交的，$\bm{\Sigma}_{\bm{f}}$ 是对角阵，对于每个因子 $i$，$\lambda_i = 0$ 当且仅当 $b_i = 0$；当因子之间具有相关性，$\lambda_i$ 可以不为 $ 0$ 即使 $b_i = 0$，即因子 $i$ 的风险溢价可能来自于某个构成 SDF 的因子 $j$。

> [!TIP|label:思考]
> 当然，同样也可能存在 $\lambda_i = 0$ 但 $b_i \neq 0$ 的情况，比如意淫一个例子：存货周转率因子没有风险溢价，即与 SDF 不相关，但如果控制市值，这个因子又与 SDF 相关了（在大公司和小公司中有相反的效应），这可能导致非 0 的风险价格。目前查阅的文献当中没有出现此类情况的说明，如果出现这样的情况意味着什么呢？

$\eqref{10}$ 式还可以被写成

$$
\begin{align}
\bm{\lambda} &= \E(\bm{f}^{\top} \bm{f}) \bm{b} = \E[\bm{f}^{\top} (\bm{f} \bm{b})] = - \E(\bm{f}^{\top} \bm{m}) \label{11} \\
\end{align}
$$

其中 $\bm{m} = (m_1,\ m_2,\ \cdots,\ m_{T})^{\top}$ 由不同时刻的 SDF 组成。

$\eqref{11}$ 式的意思是：**因子的风险溢价是因子“价格”的相反数**。如果一个因子的风险溢价是正的，那么它的价格理应是负的，因为我们假设了因子的期望为 $0$。

因此，从这个角度来说，**风险溢价捕捉的是因子是否被定价，而风险价格捕捉的是因子在定价中的贡献。**

> $\lambda_j$ captures whether factor $f_j$ is priced. ... $b_j$ captures whether $f_j$ is marginally useful in pricing assets, given the presence of other factors. &mdash;Cochrane (2009, p.242)

由于我们假设了因子的期望为 $0$，$\eqref{11}$ 式还可以被表示成

$$
\begin{equation}
\bm{\lambda} = - \Cov(\bm{f},\ \bm{m}) \\
\end{equation}
$$

对于每个因子 $i$，我们有 $\lambda_i = - \Cov(\bm{f}_{\! i},\ \bm{m})$，代表将 SDF **单独回归**到因子 $i$ 上的回归系数（和回归系数成比例）。因此**当我们看一个因子的风险溢价是否为 $0$ 时，实际上是在看它是否与 SDF 相关。**

而从 $\eqref{1}$ 式我们可以看出，$\bm{b}$ 代表着将 SDF **多元回归**到所有因子上的回归系数（和回归系数成比例）。因此**当我们看一个因子的风险价格是否为 $0$ 时，实际上在看有其他因子的情况下是否仍应该囊括这个因子。**

> $\lambda_j$ is proportional to the **single regression** coefficient of $m$ on $f_j$. $\lambda_j = 0$ asks the corresponding single regression coefficient question&mdash;"is factor $j$ correlated with the true discount factor?"
> 
> $b_j$ is the **multiple regression** coefficient of $m$ on $f_j$ given all the other factors. ... When you want to ask the question, "should I include factor $j$ given the other factors?" you want to ask the multiple regression question. &mdash;Cochrane (2009, p.242)

> [!TIP|label:单变量回归与多元回归的系数区别]
> 单变量回归是 $\Cov(\bm{f},\ \bm{m})$ 中每个因子的 Cov 除上 $\Var(\bm{f})$ 中的对角线元素；而多元回归是 $\Cov(\bm{f},\ \bm{m})$ 乘上 $[\Var(\bm{f})]^{-1}$，是对所有因子的 Cov 做了不同的线性组合得到不同因子的回归系数。


### 3. 对因子模型的再学习

刚刚进入课题组的时候，快速的过了一遍量化领域的金融基础知识，当时觉得在听论文过程中差不多够用。但是当进行项目的时候才发现，那些金融知识还远远不够。以为自己理解了，做课题的时候经常又被β、b等绕进去。所以近期就仔细把多因子模型、SDF、风险溢价、风险价格之类的再次认真看了一遍，逻辑上更深的去理解理论。

> 主要参考：
>
> 李煌师兄写的 https://leetah666.github.io/Notes/#/asset_pricing/prices_of_risk_and_risk_premia
>
> Mean-Variance 模型 https://zhuanlan.zhihu.com/p/380290863
>
> 多因子模型推导 https://zhuanlan.zhihu.com/p/588502001

风险溢价：由于承担某种风险而预期获得的额外收益 $E(R_i)-R_f$。

风险价格：投资者认为每单位风险对应的期望收益率

SDF框架 中 $\mu = \Sigma_{Rf}b$ 。Σ是资产收益与市场因子的协方差代表风险估计，b是风险价格。

因子模型 $\mu = \beta \lambda$  β是资产$f_k$在某因子$f_t$上面的风险暴露。λ=因子收益-因子期望收益是风险溢价。
![image-20240307002219346](image\image-20240307002219346.png)

 $E(R)=\beta F_1+\beta F_2+.....+\beta F_n +R_f$ 

#### 最常见的FF5模型考虑了五个因子：

市场因子（MKT）：表示市场超额回报，即市场回报减去无风险利率。

规模因子（SMB）：表示小市值公司相对于大市值公司的超额回报。

价值因子（HML）：表示高账面市值比（高价值）公司相对于低账面市值比（低价值）公司的超额回报。

动量因子（MOM）：表示过去12个月表现好的股票相对于表现差的股票的超额回报。

投资因子（CMA）：表示低投资（对未来现金流的投资少）公司相对于高投资公司的超额回报。

### 4. Markowitz Mean-Variance Portfolio

假设：市场有效，信息对等；投资者偏向规避风险，追求理性决策。

$w_i$是资产权重。$R_{it}$ 是收益率，$\mu$相当于预期

风险衡量资产收益的不确定性，通常用标准差（或方差）来度量。

公式：$\sigma_i^2 = \frac{1}{T} \sum_{t=1}^{T} (R_{it} - \mu_i)^2$

协方差衡量两个资产之间的关联程度。

公式：$\sigma_{ij} = \frac{1}{T} \sum_{t=1}^{T} (R_{it} - \mu_i) (R_{jt} - \mu_j)$

投资组合的预期收益率是各资产预期收益率的加权平均。

$\mu_p = \sum_{i=1}^{N} w_i \mu_i$ 

投资组合的整体风险水平，通常用标准差（或方差）来度量。

公式:$\sigma_p^2 = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_{ij}$

Markowitz 均值-方差投资组合理论通过最大化投资组合的预期收益率，同时最小化投资组合的方差，以实现投资组合的最优化配置。(有效边界那里还没看懂)下面是公式化表达：

![image-20240307003710538](image\image-20240307003710538.png)

## 参考文献

Back, K. (Kerry). (2017). Asset pricing and portfolio choice theory (Second edition). Oxford University Press.

Cochrane, J. H. (2009). Asset pricing (Rev. ed). Princeton University Press.

Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the cross-section. Journal of Financial Economics, 135(2), 271–292. https://doi.org/10.1016/j.jfineco.2019.06.008
