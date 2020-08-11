# Methods of Multivariate Statistics

## 4.5 A Test for a Subvector

<details>
<summary>

Often observations are taken on many characteristics, but only a few may be of great importance.
</summary>
観測されたものには多くの特徴が見受けられるが、非常に重要なのはごく一部である。
</details>

<details>
<summary>

These important characteristics, which we may call $\mathbf{x}_2$ and the remainder $\mathbf{x}_1$, can be put together.
</summary>

それらの重要な特徴は、$\mathbf{x}_2$ とその残りの $\mathbf{x}_1$ とよばれ、一緒に置くことができます
</details>
<details>
<summary>

Thus we will have $n$ $iid$ observations on $\mathbf{x}_1$ and $\mathbf{x}_2$ where we assume that

$$
  \left(
    \begin{array}{c}
      \mathbf{x}_1\\
      \mathbf{x}_2
    \end{array}
  \right)
  \sim N_p
  \left(
    \begin{array}{cc}
      \Sigma_{11} & \Sigma_{12}\\
      \Sigma_{21} & \Sigma_{22}
    \end{array}
  \right)
$$

and $\mathbf{x}_1$ and $\mathbf{x}_2$ are $s$ and $t$ vectors respectively, $s + t = p$.
</summary>

したがって、 $\mathbf{x}_1$ と $\mathbf{x}_2$ に $n$ 個の $iid$ 観測があり、次のように仮定します。

$\mathbf{x}_1$ と $\mathbf{x}_2$ はそれぞれ $s×1$ と $t×1$ のベクトルで $s+t=p$ である
</details>
<details>
<summary>

We may be interested in testing the hypothesis $H:\mathbf{\mu}_2 = \mathbf{\mu}_{20}$ against the alternative $A:\mathbf{\mu}_2 \neq \mathbf{\mu}_{20}$. 
</summary>

帰無仮説が $H:\mathbf{\mu}_2 = \mathbf{\mu}_{20}$ , 対立仮説 $A:\mathbf{\mu}_2 \neq \mathbf{\mu}_{20}$ の検定を行う
</details>
<details>
<summary>

The sample mean vector and the sample covariance matrix based on $f = n-1$ degrees of freedom can be written as

$$
\overline{\mathbf{x}}=
  \left(
    \begin{array}{c}
      \overline{\mathbf{x}}_1\\
      \overline{\mathbf{x}}_2
    \end{array}
  \right)
  , \ 
  S=
  \left(
    \begin{array}{cc}
      S_{11} & S_{12}\\
      S_{21} & S_{22}
    \end{array}
  \right)
$$

respectively.
</summary>

自由度 $f=n-1$ に基づく標本平均ベクトルと標本分散共分散行列はそれぞれこのように書くことができる。
</details>
<details>
<summary>

The likelihood ratio test is based on the statistics

$$
\frac{f-t+1}{ft}n(\overline{\mathbf{x}}_2-\mathbf{\mu_{20}})^{\prime}S_{22}^{-1}(\overline{\mathbf{x}}_2-\mathbf{\mu_{20}})
$$

which has an $F$ -distribution on $t$ and $f-t+1$ degrees of freedom.
</summary>

尤度比検定は自由度 $t$ , $f-t+1$ の$F$ 分布となる以下の統計量に基づいている
</details>
<details>
<summary>

The above test procedure treats only on the $t$-components as if we have observations.
</summary>

上記の検定手順では、$t$成分のみを観測結果として扱います
</details>
<details>
<summary>

Thus, the test procedure and confidence intervals are obtained from the marginal distribution of $\mathbf{x}_2\sim N_t(\mathbf{\mu_2,\ Σ_{22}})$ , ignoring all the information from $\mathbf{x}_1$, that is, $(\overline{\mathbf{x}}_1,\ S_{11},\ S_{12})$ .
</summary>

したがって、検定手順と信頼区間は、$\mathbf{x}_2$ の周辺分布から取得され、$\mathbf{x}_1$ からのすべての情報、つまり $(\overline{\mathbf{x}}_1,\ S_{11},\ S_{12})$ を無視します。
</details>
<details>
<summary>

This situation changes if we know the value of the mean vector, $\mathbf{\mu}_1$ , say, $\mathbf{\mu}_1 = \mathbf{\mu}_{10}$ .
</summary>

この状況では、平均ベクトル $\mathbf{\mu}_1$、たとえば $\mathbf{\mu}_1 = \mathbf{\mu}_{10}$ の値がわかっている場合に変化します。
</details>
<details>
<summary>

Then, since $\overline{\mathbf{x}}\sim N_p(\mathbf{\mu},\ n^{-1}\Sigma)$, the conditional distribution of $\overline{\mathbf{x}}_2$ , given$\overline{\mathbf{x}}_1$ , is normal with the mean $\mathbf{\mu}_2 + \Sigma_{12}^{\prime}\Sigma_{11}^{-1}(\overline{\mathbf{x}}_1 - \mathbf{\mu}_{10})$ and covariance $n^{-1}\Sigma_{2.1} = n^{-1}(\Sigma_{22}-\Sigma_{12}^{\prime}\Sigma_{11}^{-1}\Sigma_{12})$ .
</summary>

このとき、$\overline{\mathbf{x}}\sim N_p(\mathbf{\mu},\ n^{-1}\Sigma)$なので、 $\overline{\mathbf{x}}_1$ が与えられたときの $\overline{\mathbf{x}}_2$ の条件付き分布は、平均 $\mathbf{\mu}_2 + \Sigma_{12}^{\prime}\Sigma_{11}^{-1}(\overline{\mathbf{x}}_1 - \mathbf{\mu}_{10})$ 分散共分散行列 $n^{-1}\Sigma_{2.1} = n^{-1}(\Sigma_{22}-\Sigma_{12}^{\prime}\Sigma_{11}^{-1}\Sigma_{12})$ の正規分布にしたがう。
</details>
<details>
<summary>

Since $\mathbf{\mu}_{10}$ is known and $\Sigma_{12}^{\prime}\Sigma_{11}^{-1}$ can be estimated, we can use the information available from $\mathbf{x}_1$ to estimate $\mathbf{\mu}_2$ .
</summary>

$\mathbf{\mu}_{10}$ が既知で $\Sigma_{12}^{\prime}\Sigma_{11}^{-1}$ が推定できるので、 $\mathbf{x}_1$ から得られた情報を使用して $\mathbf{\mu}_2$ を推定できる。
</details>
<details>
<summary>

That is, we can use $\overline{\mathbf{x}}_2 - S_{12}^{\prime}S_{11}^{-1}(\overline{\mathbf{x}}_1 + \mathbf{\mu}_{10})$ as an estimate of $\mathbf{\mu}_2$ .
</summary>

つまり、$\overline{\mathbf{x}}_2 - S_{12}^{\prime}S_{11}^{-1}(\overline{\mathbf{x}}_1 + \mathbf{\mu}_{10})$ を $\mathbf{\mu}_2$ の推定値として使用できる。
</details>
<details>
<summary>

The situation where $\mathbf{\mu}_1$ may be known is not rare.
</summary>

$\mathbf{\mu}_1$ が知られている状況は、まれではありません。
</details>
<details>
<summary>

For example, it may be inexpensive to obtain a large number of observations on a subvector so that the corresponding population values may be assumed to be known. 
</summary>

たとえば、対応する母集団の値が既知であると仮定できるように、サブベクトルで多数の観測値を取得するのは容易な場合があります。
</details>
<details>
<summary>

In a two-sample problem, rats may have their weights recorded each week under two diets. 
</summary>

2標本問題では、ラットの体重を2つの食餌で毎週記録したものを用います。
</details>
<details>
<summary>

Assuming that the rats were randomly assigned to the two diets, the expected values of their initial weight should be the same for both groups. 
</summary>

ラットが2つの食餌が無作為に割り当てられたと仮定すると、それらの初期体重の期待値は両方のグループで同じでなければなりません。
</details>
<details>
<summary>

We now discuss the one- and two-sample cases.
</summary>

ここで、1標本と2標本の場合について議論します。
</details>

### 4.5.1 One-Sample Case

<details>
<summary>

Let $\mathbf{x}\sim N_p(\mathbf{\mu},\ \Sigma)$, $\mathbf{\mu}^{\prime}=(\mathbf{\mu}_1^{\prime},\ \mathbf{\mu}_2^{\prime})$ , and $\mathbf{x}^{\prime}=(\mathbf{x}_1^{\prime},\ \mathbf{x}_2^{\prime})$ , where $\mathbf{\mu}_1$ and $\mathbf{x}_1$ are $s$ -vectors and $\mathbf{\mu}_2$ and $\mathbf{x}_2$ are $t$ -vectors, $s+t=p$ .
</summary>

$\mathbf{x}\sim N_p(\mathbf{\mu},\ \Sigma)$, $\mathbf{\mu}^{\prime}=(\mathbf{\mu}_1^{\prime},\ \mathbf{\mu}_2^{\prime})$ , $\mathbf{x}^{\prime}=(\mathbf{x}_1^{\prime},\ \mathbf{x}_2^{\prime})$ とおく。このとき、$\mathbf{\mu}_1$ と $\mathbf{x}_1$ は $s\times 1$ , $\mathbf{\mu}_2$ と $\mathbf{x}_2$ は $t\times 1$ のベクトルで $s+t=p$ とする。
</details>
<details>
<summary>

Suppose $n$ observations are taken and the sample mean and covariance are given by $\overline{\mathbf{x}}$ and $S$.
</summary>

$n$個の観測値の平均を $\overline{\mathbf{x}}$ 分散共分散行列を $S$ とする。
</details>
<details>
<summary>

Then we partition $\overline{\mathbf{x}}$ and $S$ as

$$
\overline{\mathbf{x}}=
  \left(
    \begin{array}{c}
      \overline{\mathbf{x}}_1\\
      \overline{\mathbf{x}}_2
    \end{array}
  \right)
  , \ 
  S=
  \left(
    \begin{array}{cc}
      S_{11} & S_{12}\\
      S_{21} & S_{22}
    \end{array}
  \right)
$$

where $\overline{\mathbf{x}}_1$ is an $s$-vector and $S_{11}$ is an $s\times s$ symmetric matrix.
</summary>

このとき、$\overline{\mathbf{x}}$ と $S$ をこのように分割する。

$\overline{\mathbf{x}}_1:s\times1,\ S_{11}:s\times s$ 対称行列
</details>
<details>
<summary>

We wish to test the hypothesis

$$
H:\mathbf{\mu}_2 = \mathbf{\mu}_{20},\ \rm{given}\ \mathbf{\mu}_1 = \mathbf{\mu}_{10}\quad vs.\quad A:\mathbf{\mu}_2 \neq \mathbf{\mu}_{20},\ \rm{given}\ \mathbf{\mu}_1 = \mathbf{\mu}_{10}
$$
</summary>

帰無仮説を $\mathbf{\mu}_1 = \mathbf{\mu}_{10}$ が与えられたときの $\mathbf{\mu}_2 = \mathbf{\mu}_{20}$ , 対立仮説を $\mathbf{\mu}_1 = \mathbf{\mu}_{10}$ が与えられたときの $\mathbf{\mu}_2 \neq \mathbf{\mu}_{20}$ としたときの検定をする。
</details>
<details>
<summary>

We first define $T_p^2$ to be the usual test $H:\mathbf{\mu} = \mathbf{\mu}_{0}\quad \rm{vs.}\quad A:\mathbf{\mu} \neq \mathbf{\mu}_{0}$ , where $\mathbf{\mu}_0^{\prime}=(\mathbf{\mu}_{10}^{\prime},\ \mathbf{\mu}_{20}^{\prime})$ .
</summary>

まずはじめに、帰無仮説が $\mathbf{\mu} = \mathbf{\mu}_{0}$ , 対立仮説は $\mathbf{\mu} \neq \mathbf{\mu}_{0}$ となるような $T_p^2$ を定義する。
</details>
<details>
<summary>

That is,

$$
T_p^2 = n(\overline{\mathbf{x}}-\mathbf{\mu}_0)^{\prime}S^{-1}(\overline{\mathbf{x}}-\mathbf{\mu}_0) \tag{4.5.1}
$$
</summary>

それはこのように定義される。
</details>
<details>
<summary>

We also define $T_s^2$ to be the $T^2$ -statistic for testing $H:\mathbf{\mu}_1 = \mathbf{\mu}_{10}\quad \rm{vs.}\quad A:\mathbf{\mu}_1 \neq \mathbf{\mu}_{10}$ .
</summary>

また、帰無仮説が $\mathbf{\mu}_1 = \mathbf{\mu}_{10}$ , 対立仮説は $\mathbf{\mu}_1 \neq \mathbf{\mu}_{10}$ , 検定統計量が $T^2$ となる $T_s^2$ を定義する。
</details>
<details>
<summary>

That is,

$$
T_s^2 = n(\overline{\mathbf{x}}_1-\mathbf{\mu}_{10})^{\prime}S_{11}^{-1}(\overline{\mathbf{x}}_1-\mathbf{\mu}_{10}) \tag{4.5.2}
$$
</summary>

それはこのように定義される。
</details>
<details>
<summary>

Then we reject $H:\mathbf{\mu}_2 = \mathbf{\mu}_{20},\ \rm{given}\ \mathbf{\mu}_1 = \mathbf{\mu}_{10}$, if

$$
\frac{f-p+1}{t}\frac{T_p^2-T_s^2}{f+T_s^2}>F_{t,\ f-p+1,\ \alpha}\tag{4.5.3}
$$
</summary>

このとき、この不等式を満たしたら帰無仮説 $H:\mathbf{\mu}_2 = \mathbf{\mu}_{20},\ \rm{given}\ \mathbf{\mu}_1 = \mathbf{\mu}_{10}$ を棄却する。
</details>
<details>
<summary>

This is the likelihood ratio test, and the same test is obtained even if $\Sigma_{11}$ is known.
</summary>

これは尤度比検定であり、$\Sigma_ {11}$ が既知でも同じ検定が得られます。
</details>
<details>
<summary>

This problem has been considered by Rao (1949), Olkin and Shrikhande (1953), and Giri (1964).
</summary>

この問題は、Rao (1949), Olkin and Shrikhande (1953), Giri (1964) でも考えられています。
</details>
<details>
<summary>

To find a confidence interval for $\mathbf{a^{\prime}\mu_2}$ , let

$$
c_{\alpha}^2=t(f-p+1)^{-1}F_{t,\ f-p+1,\ \alpha}\ ,
$$
$$
S_{2.1}=S_{22}-S_{12}^{\prime}S_{11}^{-1}S_{12}.
$$
</summary>

$\mathbf{a^{\prime}\mu_2}$ の信頼区間を調べるために、$c_{\alpha}^2$ と $S_{2.1}$ をこのようにおく。
</details>
<details>
<summary>

Then a $(1-\alpha)100\%$ confidence interval for $\mathbf{a^{\prime}\mu_2}$ is given by

$$
\mathbf{a}[\overline{\mathbf{x}}_2 + S_{12}^{\prime}S_{11}^{-1}(\overline{\mathbf{x}}_1 - \mathbf{\mu}_{10})]\pm n^{-1/2}[f + T_s^{2}]^{1/2}c_{\alpha}(\mathbf{a}^{\prime}S_{2.1}\mathbf{a})^{1/2}.\tag{4.5.4}
$$
</summary>

このとき、$\mathbf{a^{\prime}\mu_2}$ の $(1-\alpha)100\%$ 信頼区間はこのようになる。
</details>
<details>
<summary>

If our interest is in only a small number, k, of confidence intervals, we should check if

$$
(f-s)^{-1}t_{f-s,\ \frac{\alpha}{2k}}^2\leq c_{\alpha}^2.
$$
</summary>

関心がk(small number)のときの信頼区間のみにある場合、この不等式が成り立っているかを確認する必要があります。
</details>
<details>
<summary>

If this is the case, then $c_{\alpha}$ should be replaced by $(f-s)^{\frac{1}{2}}t_{f-s,\ \frac{\alpha}{2k}}$ to obtain shorter Bonferroni confidence intervals
</summary>

この場合、$c_{\alpha}$ を $(f-s)^{\frac{1}{2}}t_{f-s,\ \frac{\alpha}{2k}}$に置き換えて、 より短いBonferroni信頼区間をえる。
</details>

<u>Example 4.5.1</u> 
<details>
<summary>

The following artificial example fives the scores of 10 randomly selected students on an achievement test.
</summary>

次の人工的な例は、無作為に選ばれた10人の生徒の成績テストのスコアを示しています。
</details>
<details>
<summary>

It is known that the mean for all students writing the test was 50.
</summary>

テストを受けたすべての学生の平均が50点であった。
</details>
<details>
<summary>

The subjects underwent training for a similar test given nationally.
</summary>

被験者は全国的に行われている同様のテストの訓練を受けました。
</details>
<details>
<summary>

From the data in Table 4.5.1, the sample covariance and correlation matrices are

$$
S=\left(
    \begin{array}{cc}
      250 & 159\\
      159 & 148
    \end{array}
  \right)
  \quad
  \rm{and}
  \quad
R=\left(
    \begin{array}{cc}
      1 & 0.83\\
      0.83 & 1
    \end{array}
  \right)
$$

respectively.
</summary>

表4.5.1のデータから、サンプルの分散共分散行列と相関行列はそれぞれ $S$ と $R$ です。
</details>
<details>
<summary>

Note there is a high correlation (0.83) between the two test scores.
</summary>

2つのテストスコアの間に高い相関（0.83）があることに注意してください。
</details>
<details>
<summary>

$\rm{Table\ 4.5.1}$
</summary>

|               | Before Training| After Training |
|:-------------:|:--------------:|:--------------:|
|               | 70             | 75             |
|               | 60             | 58             |
|               | 65             | 70             |
|               | 50             | 55             |
|               | 43             | 48             |
|               | 40             | 41             |
|               | 80             | 78             |
|               | 45             | 65             |
|               | 30             | 55             |
|               | 40             | 50             |
| Sample mean   | 52.3           |59.5            |
|Population mean|$\mu_1$         |$\mu_2$         |
</details>
<details>
<summary>

To test whether training affects average score, we test the hypothesis

$$H:\mu_1=50,\ \mu_2=50\quad \rm{vs.}\quad A:\mu_1=50,\ \mu_2\neq50$$
</summary>

トレーニングが平均スコアに影響するかどうかをテストするために、この仮説を検定します。
</details>
<details>
<summary>

The chosen value $\mu_2$ was the mean nationally for all students taking the test.
</summary>

選択した値 $\mu_2$ は、テストを受けるすべての学生の全国的な平均値です。
</details>
<details>
<summary>

Calculations yield that $T_2^2=14.1,\ f=10-1=9,\ s=1,\ t=1,\ p=2,\ T_1^2=0.21$, and

$$
\begin{aligned}
  \frac{f-p+1}{t}\frac{T_2^2-T_1^2}{f+T_1^2}&=\frac{8}{1}\times\frac{13.99-0.21}{9+0.21}\\
  &=12.03\\
  &\geq F_{1,\ 8,\ 0.05}\\
  &=5.32
\end{aligned}
$$
</summary>

計算により、このようになる。
</details>
<details>
<summary>

We conclude that the training has an effect on the scores of the students. 
</summary>

トレーニングは生徒のスコアに影響を与えると結論付けています。
</details>
<details>
<summary>

To find a confidence interval for $\mu_2$, we compute

$$
\begin{aligned}
  c_{\alpha}^2 &= \frac{t}{f-p+1}F_{1,\ 8,\ 0.05} = \frac{1}{8}\times 5.32 = 0.665\\
  S_{21}S_{11}^{-1}&=159/250=0.636\\
  S_{2.1}&=S_{22}-S_{21}S_{11}^{-1}S_{12}=148-(159)^2/250=46.876\\
  z&=\overline{x}_1-\mu_{10}=52.3-50=2.3
\end{aligned}
$$
</summary>

$\mu_2$ の信頼区間を見つけるには、これらを計算する。
</details>
<details>
<summary>

Note that in this case, $f-s=f-p+1=8$ and $t=1$.
</summary>

このとき、 $f-s=f-p+1=8$ かつ $t=1$ に注意する。
</details>
<details>
<summary>

Since we are interested in only one confidence interval, the confidence interval obtained by the Bonferroni bound will be the same.
</summary>

調べたい信頼区間は1つだけなので、Bonferroniの信頼区間の範囲と同じになります。
</details>
<details>
<summary>

Note further that in this case $(f-p+1)^{-1}tF_{1,\ f-p+1,\ .05}=(f-s)^{-1}t^2_{f-s,\ .025}$.
</summary>

さらに、この場合は $(f-p+1)^{-1}tF_{1,\ f-p+1,\ .05}=(f-s)^{-1}t^2_{f-s,\ .025}$ となる。
</details>
<details>
<summary>

Thus, a $95\%$ confidence interval for $\mu_2$ is given by

$$
59.5-0.636\times2.3\pm(10)^{-\frac{1}{2}}(9+0.21)^{\frac{1}{2}}(0.665)^{\frac{1}{2}}(45.876)^{\frac{1}{2}}
$$

or

$$
(52.69,\ 63.38)
$$
</summary>

$\mu_2$ の $95\%$ 信頼区間はこのようになる。
</details>

<details>

<summary>

Example 4.5.1 Subvector Test (one sample)
python
</summary>

```python
# ----------Example 4.5.1------------
# ----Subvector test (one sample)----
import math
import numpy as np
from scipy.stats import f as fdp
from scipy.stats import t as tdp
# x_1, x_2, x, x_bar
x1 = np.array([70, 60, 65, 50, 43, 40, 80, 45, 30, 40])
x2 = np.array([75, 58, 70, 55, 48, 41, 78, 65, 55, 50])
x = np.col_stack((x1, x2))
xb = np.array([np.mean(x1), np.mean(x2)])
# n, p, s, t, f, μ_0, α
n, p, s, t = 10, 2, 1, 1
f = n - 1
mu0 = np.array([50, 50])
alpha = 0.05
# S bias
S = np.cov(xx, rowvar=1, bias=0)
S_inv = np.linalg.inv(S)
# Tpsq, Tssq
Tpsq = n*(xb-mu0).T @ S_inv @ (xb-mu0)
Tssq = n*(xb[0]-mu0[0])*(1/S[0,0])*(xb[0]-mu0[0])
# F0, Fa
F0 = (f-p+1)/t*(Tpsq-Tssq)/(f+Tssq)
Fa = fdp.ppf(1-alpha, t, f-p+1)
if F0 > Fa:
    print("F0 > Fa: reject")
else:
    print("F0 < Fa: reject")
casq = t / (f-p+1) * Fa
S21 = S[1, 1] - S[1, 0] ** 2 * 1 / S[0, 0]
tt = 1 / (f - s) * tdp.ppf(1 - alpha / 2, f - s) ** 2
if tt < casq:
    c = tt
else:
    c = casq
low = xb[1] - S[0, 1] * (1 / S[0, 0]) * (xb[0] - mu0[0]) - (1 / math.sqrt(n)) * math.sqrt(f + Tssq) * math.sqrt(c) * math.sqrt(S21)
upp = xb[1] - S[0, 1] * (1 / S[0, 0]) * (xb[0] - mu0[0]) + (1 / math.sqrt(n)) * math.sqrt(f + Tssq) * math.sqrt(c) * math.sqrt(S21)
"({:.6f}, {:.6f})".format(low, upp)


```

</details>

### 4.5.2 Two-Sample Case

<details>
<summary>

In the two-sample problem we obtain the two-sample means $\overline{\mathbf{x}}$ and $\overline{\mathbf{y}}$, based on $n_1$ and $n_2$ observations.
</summary>

2標本問題では、$n_1$ , $n_2$ 個の観測に基づいて、2標本平均 $\overline{\mathbf{x}}$ と $\overline{\mathbf{y}}$ をえる。
</details>
<details>
<summary>

That is, $\overline{\mathbf{x}}\sim N_p(\mathbf{\mu},\ \Sigma/n_1)$ and $\overline{\mathbf{y}}\sim N_p(\mathbf{\gamma},\ \Sigma/n_2)$.
</summary>

$\overline{\mathbf{x}}$ と $\overline{\mathbf{y}}$ の分布はこのようになる。
</details>
<details>
<summary>

We also obtain $S$, the pooled estimate of $\Sigma$, on $f=n_1+n_2-2$ degrees of freedom.
</summary>

自由度 $f$ で $S$、$\Sigma$ のプールされた推定値も取得します。（不偏）
</details>
<details>
<summary>

We partition $\overline{\mathbf{x}}$, $\overline{\mathbf{y}}$, $\mathbf{\mu}$, $\mathbf{\gamma}$, and $S$ as

$$
\overline{\mathbf{x}}=
  \left(
    \begin{array}{c}
      \overline{\mathbf{x}}_1\\
      \overline{\mathbf{x}}_2
    \end{array}
  \right)
  ,
  \quad
\overline{\mathbf{y}}=
  \left(
    \begin{array}{c}
      \overline{\mathbf{y}}_1\\
      \overline{\mathbf{y}}_2
    \end{array}
  \right)
  ,
  \quad
\mathbf{\mu}=
  \left(
    \begin{array}{c}
      \mathbf{\mu}_1\\
      \mathbf{\mu}_2
    \end{array}
  \right)
  ,
  \quad
\mathbf{\gamma}=
  \left(
    \begin{array}{c}
      \mathbf{\gamma}_1\\
      \mathbf{\gamma}_2
    \end{array}
  \right)
  ,
  \quad
  S=
  \left(
    \begin{array}{cc}
      S_{11} & S_{12}\\
      S_{21} & S_{22}
    \end{array}
  \right)
$$

where $\overline{\mathbf{x}}_1$, $\overline{\mathbf{y}}_1$, $\mathbf{\mu}_1$ and $\mathbf{\gamma}_1$ are $s$-vectors and $S_{11}$ is an $s\times s$ matrix; $\overline{\mathbf{x}}_2$, $\overline{\mathbf{y}}_2$, $\mathbf{\mu}_2$ and $\mathbf{\gamma}_2$ are $t$-vectors, and $S_{22}$ is an $t\times t$ matrix, where $s+t=p$.
</summary>

$\overline{\mathbf{x}}$, $\overline{\mathbf{y}}$, $\mathbf{\mu}$, $\mathbf{\gamma}$, と $S$ をこのように分割する。
</details>
<details>
<summary>

Then the hypothesis

$$
H:\mathbf{\mu}_2 = \mathbf{\gamma}_{2}\quad\rm{given}\quad\mathbf{\mu}_1 = \mathbf{\gamma}_{1}\quad vs.\quad A:\mathbf{\mu}_2 \neq \mathbf{\gamma}_{2}\quad\rm{given}\quad\mathbf{\mu}_1 = \mathbf{\gamma}_{1}
$$

is rejected if

$$
\frac{f-p+1}{t}\frac{T_p^2-T_s^2}{f+T_s^2}\geq F_{t,\ f-p+1,\ \alpha}\tag{4.5.5}
$$

where

$$
T_p^2 = \frac{n_1n_2}{n_1+n_2}(\overline{\mathbf{x}}-\overline{\mathbf{y}})^{\prime}S^{-1}(\overline{\mathbf{x}}-\overline{\mathbf{y}})
$$

$$
T_s^2 = \frac{n_1n_2}{n_1+n_2}(\overline{\mathbf{x}}_1-\overline{\mathbf{y}}_1)^{\prime}S_{11}^{-1}(\overline{\mathbf{x}}_1-\overline{\mathbf{y}}_1)
$$

and $f=n_1+n_2-2$.
</summary>

このとき、帰無仮説と対立仮説をこのようにおくと、$(4.5.5)$ の不等式が成り立ったら帰無仮説を棄却する。
</details>
<details>
<summary>

A $(1-\alpha)100\%$ confidence interval for $\mathbf{a}^{\prime}(\mathbf{\mu}_2-\mathbf{\gamma}_2)$ is given by

$$
\mathbf{a}[\overline{\mathbf{x}}_2 - \overline{\mathbf{y}}_2 + S_{21}S_{11}^{-1}(\overline{\mathbf{x}}_1 - \overline{\mathbf{y}}_1)]\pm b^{-1}[f + T_s^{2}]^{1/2}c_{\alpha}(\mathbf{a}^{\prime}S_{2.1}\mathbf{a})^{1/2}.\tag{4.5.6}
$$

where

$$
f=n_1+n_2-2\quad b^2=n_1n_2/(n_1+n_2)\quad c^2_{\alpha}=t(f-p+1)^{-1}F_{t,\ f-p+1,\ \alpha}
$$
</summary>

$\mathbf{a}^{\prime}(\mathbf{\mu}_2-\mathbf{\gamma}_2)$ の $(1-\alpha)100\%$ 信頼区間はこのようになる。
</details>
<details>
<summary>

For $k$ confidence intervals, Bonferroni\primes inequality enables us to replace $c^2_{\alpha}$ with $(f-s)^{-1}t^2_{f-s,\ \frac{\alpha}{2k}}$, if the latter is smaller.
</summary>

$k$ 信頼区間の場合、Bonferroniの不等式により、後者が小さい場合$ c^2_{\alpha}$ を $(f-s)^{-1}t^2_{f-s,\ \frac{\alpha}{2k}}$ で置き換えることができます。
</details>

<u>Example 4.5.2</u>

<details>
<summary>

Table 4.5.2 shows gains in fish length over a two-week period under a standard and a test diet.
</summary>

表4.5.2は、標準食と試験食での2週間にわたる魚の長さの増加を示しています。
</details>
<details>
<summary>

As the fish were randomly assigned to the diets, the expected initial lengths of the two groups are the same.
</summary>

魚をランダムに飼料を割り当てたとき、2つのグループの予想される初期の長さは同じです。
</details>
<details>
<summary>

We now wish to test for the difference in gain in length for the groups.
</summary>

ここで、2グループの長さの増加の違いを検定します。
</details>
<details>
<summary>

That is, we test

$$
H:
\left(
  \begin{array}{c}
    \mu_1\\
    \mu_2\\
    \mu_3
  \end{array}
\right)=
\left(
  \begin{array}{c}
    \gamma_1\\
    \gamma_2\\
    \gamma_3
  \end{array}
\right)
\quad
\rm{vs.}
\quad
A:\mu_1=\gamma_1,\quad\mu_i=\gamma_i,\quad i=2,\ 3
$$
</summary>

そのために、この検定を行う。
</details>
<details>
<summary>

$\rm{Table 4.5.2}$
</summary>

<table>
	<tr>
		<td></td>
		<td colspan="3">Standard Diet</td>
		<td colspan="3">Test Diet</td>
	</tr>
	<tr>
		<td></td>
		<td noWrap>Initial Length</td>
		<td>Week1</td>
		<td>Week2</td>
		<td>Initial Length</td>
		<td>Week1</td>
		<td>Week2</td>
	</tr>
	<tr>
		<td></td>
		<td>12.3</td>
		<td>2.5</td>
		<td>2.9</td>
		<td>12.0</td>
		<td>2.3</td>
		<td>2.7</td>
	</tr>
	<tr>
		<td></td>
		<td>12.1</td>
		<td>2.2</td>
		<td>2.5</td>
		<td>11.8</td>
		<td>2.0</td>
		<td>2.4</td>
	</tr>
	<tr>
		<td></td>
		<td>12.8</td>
		<td>2.9</td>
		<td>3.0</td>
		<td>12.7</td>
		<td>3.1</td>
		<td>3.6</td>
	</tr>
	<tr>
		<td></td>
		<td>12.0</td>
		<td>2.1</td>
		<td>2.2</td>
		<td>12.4</td>
		<td>2.8</td>
		<td>3.2</td>
	</tr>
	<tr>
		<td></td>
		<td>12.1</td>
		<td>2.2</td>
		<td>2.4</td>
		<td>12.1</td>
		<td>2.5</td>
		<td>2.8</td>
	</tr>
	<tr>
		<td></td>
		<td>11.8</td>
		<td>1.9</td>
		<td>2.0</td>
		<td>12.0</td>
		<td>2.2</td>
		<td>2.7</td>
	</tr>
	<tr>
		<td></td>
		<td>12.7</td>
		<td>2.9</td>
		<td>3.3</td>
		<td>11.7</td>
		<td>2.0</td>
		<td>2.4</td>
	</tr>
	<tr>
		<td></td>
		<td>12.5</td>
		<td>2.7</td>
		<td>3.0</td>
		<td>12.2</td>
		<td>2.5</td>
		<td>3.0</td>
	</tr>
	<tr>
		<td>Sample Means</td>
		<td>12.2875</td>
		<td>2.425</td>
		<td>2.6625</td>
		<td>12.1125</td>
		<td>2.425</td>
		<td>2.85</td>
	</tr>
	<tr>
		<td>Population Means</td>
		<td>
      $\mu_1$
    </td>
		<td>$\mu_2$</td>
		<td>$\mu_3$</td>
		<td>$\gamma_1$</td>
		<td>$\gamma_2$</td>
		<td>$\gamma_3$</td>
	</tr>
</table>

</details>
<details>
<summary>

The pooled covariance matrix is given by

$$
S=\left(
    \begin{array}{ccc}
      0.114 & 0.128 & 0.140\\
      0.128 & 0.146 & 0.161\\
      0.140 & 0.161 & 0.186
    \end{array}
  \right)
  \quad
S^{-1}=\left(
    \begin{array}{rrr}
      405.636 & -358.155 & 3.6168\\
      -358.155 & 447.966 & -117.122\\
      3.6168 & -117.122 & 103.950
    \end{array}
  \right)
$$
</summary>

プールされた分散共分散行列とその逆行列はこのように計算できる。
</details>
<details>
<summary>

Calculations yield $T^2_1=1.07,\ f=8+8-2=14,\ T_3^2=63.36,\ s=1,\ t=2, p=3,\ b^2=8\times8/(8+8)=4,\ f-s=13$, and

$$
\begin{aligned}
  F_{2,\ 12}&=\frac{12}{2}\frac{63.36-1.07}{14+1.07}=24.793\\
  &\geq F_{2,\ 12,\ 0.05}=3.89
\end{aligned}
$$
</summary>

計算結果はこのようになる。
</details>
<details>
<summary>

Hence we reject $H$ at $\alpha=0.05$ and claim that the test diet differs from the standard diet with respect to the effect on gain in length.
</summary>

したがって、$\alpha = 0.05$ で帰無仮説を棄却し、飼食が長さの増加への影響に関して標準ダイエットとは異なると主張します。
</details>
<details>
<summary>

To obtain joint $95\%$ confidence intervals for $\mu_2-\gamma_2$ and $\mu_3-\gamma_3$, we compute

$$
c^2_{\alpha}=\frac{t}{f-P+1}F_{1,\ f-p+1,\ .05}=\frac{2}{12}\times3.89=0.648
$$
$$
(f-s)^{-1}t^2_{f-s,\ \alpha/4}=0.493404
$$
</summary>

$95\%$ の $\mu_2-\gamma_2$ と $\mu_3-\gamma_3$ の結合信頼区間を求めるためにこれを計算する。
</details>
<details>
<summary>

Since $c_{\alpha}^2>(f-s)^{-1}t^2_{f-s,\ \alpha/4}$, we shall use Bonferroni bounds to obtain confidence intervals for $\mu_2-\gamma_2$ and $\mu_3-\gamma_3$.
</summary>

$c_{\alpha}^2>(f-s)^{-1}t^2_{f-s,\ \alpha/4}$ なので、Bonferroniの範囲を使用して$\mu_2-\gamma_2$ と $\mu_3-\gamma_3$ の信頼区間を取得する。
</details>
<details>
<summary>

Since,

$$
\begin{aligned}
  b^2&=\frac{8\times8}{8+8}=4\\
  f-s&=14-1=13\\
  S_{21}S_{11}^{-1}&=(0.114)^{-1}
  \left(
    \begin{array}{c}
      0.128\\
      0.140
    \end{array}
  \right)=
  \left(
    \begin{array}{c}
      1.1205\\
      1.2277
    \end{array}
  \right)\\
  S_{2.1}&=S_{22}-S_{21}S_{11}^{-1}S_{12}\\
  &=
  \left(
    \begin{array}{cc}
      0.146 & 0.161\\
      0.161 & 0.186
    \end{array}
  \right)-
  \left(
    \begin{array}{c}
      1.1205\\
      1.2277
    \end{array}
  \right)
  (0.128, 0.140)\\
  &=
  \left(
    \begin{array}{cl}
      .003165 & .003566\\
      .003566 & .01364
    \end{array}
  \right)
\end{aligned}
$$
</summary>

よってこうなる。
</details>
<details>
<summary>

The difference between the sample means of the initial length = 12.2875 $-$ 12.1125 = 0.1750, and the $95\%$ confidence intervals for linear combinations of $(\mu_2-\gamma_2,\ \mu_3-\gamma_3)$ are given by

$$
\mathbf{a}^{\prime}
\left\{
  \left(
    \begin{array}{c}
      2.425-2.425\\
      2.6625-2.85
    \end{array}
  \right)-
  \left(
    \begin{array}{c}
      1.1205\\
      1.2277
    \end{array}
  \right)(0.1750)
\right\}
\pm\frac{1}{2}(14+1.07)^{\frac{1}{2}}(0.493)^{\frac{1}{2}}
\left\{
  \mathbf{a}^{\prime}
  \left(
    \begin{array}{cl}
      .003165 & .003566\\
      .003566 & .01364
    \end{array}
  \right)
  \mathbf{a}
\right\}^{\frac{1}{2}}
$$
</summary>

魚の初期の長さの標本平均の差は0.1750で、$(\mu_2-\gamma_2,\ \mu_3-\gamma_3)$ の線形結合の $95\%$ 信頼区間はこのようになる。
</details>
<details>
<summary>

We choose $\mathbf{a}=(1,\ 0)$ and $\mathbf{a}=(0,\ 1)$, respectively, to obtain joint $95\%$ confidence intervals for $\mu_2-\gamma_2$ and $\mu_3-\gamma_3$.
</summary>

$95\%$ の $\mu_2-\gamma_2$ と $\mu_3-\gamma_3$ の結合信頼区間を得るために、$\mathbf{a}$ をそれぞれこのように選ぶ。
</details>
<details>
<summary>

They are given by

$$
\begin{aligned}
  \mu_2-\gamma_2:(-.2728,\ -.1194)\\
  \mu_3-\gamma_3:(-.5616,\ -.2431)
\end{aligned}
$$

</summary>

それはこのようになる。
</details>
<details>
<summary>

Hence the test diet and standard diet produced different length gains both weeks, and because both intervals are below zero, we can state that the gain in length of a fish is lower with the standard diet than with the new test diet.
</summary>

したがって、テストダイエットと標準ダイエットは両方の週で異なる長さの増加をもたらし、両方の間隔が0未満であるため、標準の食事では新しいテストダイエットよりも魚の長さの増加が低いと言えます。
</details>
<details>
<summary>

The variable initial length is called a concomitant variable or covariable.
</summary>

変数の初期長は、付随変数または共変数と呼ばれます。
</details>
<details>
<summary>

It should be included in the model only if it is correlated with the response variables of interest.
</summary>

対象の目的変数と相関している場合にのみ、モデルに含める必要があります。
</details>
<details>
<summary>

If the initial length of the fish did not affect the gain in length of the fish, then the addition of the covariable would only add to the variation in the experiment and decrease the power of the tests.
</summary>

魚の初期の長さが魚の長さの増加に影響しなかった場合、共変数の追加は、実験の変動に追加し、テストの検出力を低下させるだけです。
</details>

