# Theoretical Extension of *Early Failure Detection and Intervention in Video Diffusion Models*

The paper's empirical findings: (i) over $T=50$ denoising steps, the MAE between VBench scores on the L2R intermediate preview and on the final video drops sharply within the first ~10 steps and then plateaus (Fig. 10); (ii) semantic convergence is highly sample-dependent across prompts (Fig. S2 — "dog" locks in early, "shark" remains ambiguous until late steps), which is why the authors fall back to a hand-crafted sliding-window dynamic detector.

This note addresses two questions: (1) can the "early encoding" phenomenon be proved rigorously? (2) what should a prompt→convergence-speed predictive model look like?

---

## 1. Rigorous Theory for Early Semantic Encoding

**It can be proved rigorously — but only under assumptions.** A fully distribution-free, model-free proof does not exist. Once reasonable assumptions are granted, the argument proceeds in three layers.

### Layer A. Exact spectral decomposition in the linear-Gaussian case (hard theorem)

Let $z_0 \sim \mathcal{N}(0, \Sigma_0)$ (or conditionally $z_0 \mid c \sim \mathcal{N}(\mu_c, \Sigma_{0|c})$) with forward process $z_t = \alpha_t z_0 + \sigma_t \varepsilon$. The Bayes-optimal denoiser (Tweedie) is

$$\hat z_{0|t} = \mathbb{E}[z_0 \mid z_t, c] = \alpha_t \Sigma_{0|c} (\alpha_t^2 \Sigma_{0|c} + \sigma_t^2 I)^{-1} z_t + (\text{mean terms})$$

Diagonalizing $\Sigma_{0|c} = \sum_i \lambda_i v_i v_i^\top$, the per-direction **Wiener gain** is

$$g_i(t) = \frac{\alpha_t^2 \lambda_i}{\alpha_t^2 \lambda_i + \sigma_t^2} = \frac{\lambda_i \cdot \mathrm{SNR}(t)}{\lambda_i \cdot \mathrm{SNR}(t)+1}$$

and the posterior variance (MMSE) along $v_i$ is $\mathrm{Var}(z_0 \cdot v_i \mid z_t) = \lambda_i \sigma_t^2 / (\alpha_t^2\lambda_i + \sigma_t^2)$.

**Rigorous conclusion.** For any tolerance $\varepsilon$, the error along axis $i$ falls below $\varepsilon$ once $\mathrm{SNR}(t) \ge \frac{1-\varepsilon}{\varepsilon \lambda_i}$. **High-$\lambda_i$ directions (low-frequency / semantic modes) require $\lambda_{\max}/\lambda_{\min}$-times less SNR than low-$\lambda_i$ directions (high-frequency texture).** This is the first-principles origin of coarse-to-fine denoising, and therefore of "semantics are encoded first."

This layer is an **exact equality** under the linear-Gaussian assumption, not an approximation.

### Layer B. Information-theoretic bound in the general case (rigorous, but only a rate)

For arbitrary $p(z_0 \mid c)$ with finite second moments, keeping the Gaussian forward process gives

$$I(z_0; z_t \mid c) = h(z_0 \mid c) - h(z_0 \mid z_t, c)$$

Combining de Bruijn's identity with the HWI / Gaussian Poincaré inequalities yields $I$ strictly monotonic in $\mathrm{SNR}(t)$ (Guo–Shamai–Verdú 2005, I-MMSE), and

$$\frac{d\,\mathrm{MMSE}(\mathrm{SNR})}{d\,\mathrm{SNR}} \;=\; -\,\mathbb{E}\bigl\|\hat z_{0|t}(\mathrm{SNR})-z_0\bigr\|^2 \cdot (\text{const})$$

This gives a distribution-agnostic but **rate-only** guarantee: we cannot prove "step 10" specifically — the constant depends on the actual spectrum of $\Sigma_{0|c}$.

### Layer C. Propagating latent error to the alignment score (rigorous, closing the loop)

Assume the decoder $\mathcal{D}$ and the ViCLIP video encoder $f$ are $L_\mathcal{D}$- and $L_f$-Lipschitz respectively (empirically valid for pretrained networks; PAC-Bayes-type bounds are available). Then

$$|s_t - s_0^\star| \;\le\; L_f L_\mathcal{D} \cdot \|\hat z_{0|t} - z_0^\star\|_2 \;\le\; L_f L_\mathcal{D} \sqrt{\mathrm{tr}\,\Sigma_{\mathrm{post}}(t)}$$

Substituting the MMSE decay from Layers A/B produces an **analytic upper bound on the "normalized MAE vs. $t$" curve** of Fig. 10 — and this bound can be split between a semantic subspace (top-$k$ eigendirections) and a detail subspace, explaining why **Overall Consistency converges earlier than Imaging Quality** (Fig. 10 and Fig. 9 together).

### What cannot be proved

- The exact step index (e.g., "step 10") is data- and schedule-dependent. Theory only yields $t^\star \asymp T - \Theta(\log \lambda_{\max}/\lambda_{\min})$; the constants require empirical calibration.
- The network's score-approximation error $\|\epsilon_\theta - \nabla \log p_t\|$ cannot be ignored without assumptions; the standard fix is to assume an $\varepsilon$-accurate score network (cf. Chen et al. 2023 on DDPM sample complexity).
- "Failure" must be formalized as an event (e.g., $s_0^\star < \tau$), then Layer C plus a concentration inequality gives a PAC-type "early $s_k$ identifies failures with probability $1-\delta$" statement.

**Bottom line:** a rigorous proof exists, but it is the combination "spectrum + I-MMSE + Lipschitz," not a single clean theorem, and its quantitative constants must be empirically calibrated.

---

## 2. Main Result: Early Predictability of Final Quality

This section gives the formal statement that underwrites *quality-gated early stopping* (Type C) in diffusion sampling: the observable alignment score at any intermediate denoising step $t$ is an unbiased, consistent, and exponentially concentrated estimator of the final-video score.

### 2.1 Setup and assumptions

Let $(z_t)_{t \in [0,T]}$ be a latent diffusion forward process with
$$z_t = \alpha_t z_0 + \sigma_t \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,I),\qquad \mathrm{SNR}(t) := \alpha_t^2/\sigma_t^2,$$
driven by a condition $c$ (text prompt). Write $p_t(\cdot\mid c)$ for the marginal and $\mathcal{F}_t := \sigma(z_t, c)$ for the filtration generated by the state at time $t$. Denote the Bayes-optimal clean estimate by
$$\hat z_{0|t} := \mathbb{E}[z_0 \mid z_t, c].$$
Let $\mathcal{D}:\mathbb{R}^{d_z}\to\mathbb{R}^{d_x}$ be the decoder, $q:\mathbb{R}^{d_x}\to\mathbb{R}$ the quality scorer (e.g., ViCLIP alignment), and define
$$s_t := q(\mathcal{D}(\hat z_{0|t})),\qquad s_0^\star := q(\mathcal{D}(z_0^\star)),$$
where $z_0^\star$ is the clean latent produced by full denoising.

**Assumptions.**

- **(A1) Gaussian forward kernel.** The forward process is as above with a monotonically increasing schedule $\mathrm{SNR}:[0,T]\to[0,\infty)$ (decreasing in $t$).
- **(A2) Lipschitz decoder.** There exists $L_\mathcal{D}<\infty$ such that $\|\mathcal{D}(a)-\mathcal{D}(b)\|_2 \le L_\mathcal{D}\|a-b\|_2$ for all $a,b$.
- **(A3) Lipschitz scorer.** There exists $L_q<\infty$ such that $|q(x)-q(y)| \le L_q\|x-y\|_2$ for all $x,y$.
- **(A4) Finite data variance.** $\Sigma_{0|c} := \mathrm{Cov}(z_0\mid c)$ has finite trace.

Write $L := L_q L_\mathcal{D}$ and $\Sigma_{\mathrm{post}}(t) := \mathrm{Cov}(z_0 \mid z_t, c)$.

### 2.2 Preliminary lemmas

**Lemma 1** (Tweedie / MMSE identity). *Under (A1), for every $t$,*
$$\hat z_{0|t} = \frac{z_t + \sigma_t^2\,\nabla_{z_t}\log p_t(z_t\mid c)}{\alpha_t} = \arg\min_{f \in L^2(\mathcal{F}_t)} \mathbb{E}\|z_0 - f(z_t)\|^2.$$

*Proof.* The first equality is Tweedie's formula (Robbins 1956; Efron 2011) applied to the Gaussian likelihood in (A1). The second is the MMSE characterization of the conditional expectation. $\square$

**Lemma 2** (Martingale property). *The family $\{\hat z_{0|t}\}_{t \in [0,T]}$ is a martingale with respect to $(\mathcal{F}_t)$ read in the reverse (decreasing-$t$) direction; that is, for any $s\le t$,*
$$\mathbb{E}[\hat z_{0|s}\mid \mathcal{F}_t] = \hat z_{0|t}.$$

*Proof.* Since $\mathcal{F}_s \supseteq \mathcal{F}_t$ for $s\le t$ (smaller noise, more information), the tower property gives
$$\mathbb{E}[\hat z_{0|s}\mid \mathcal{F}_t] = \mathbb{E}\bigl[\mathbb{E}[z_0\mid\mathcal{F}_s]\,\big|\,\mathcal{F}_t\bigr] = \mathbb{E}[z_0\mid\mathcal{F}_t] = \hat z_{0|t}. \qquad \square$$

**Lemma 3** (Posterior-variance decay). *Under (A1, A4), $\mathrm{tr}\,\Sigma_{\mathrm{post}}(t)$ is monotonically non-increasing as $\mathrm{SNR}(t)$ grows (i.e., as $t$ decreases). If in addition $p(z_0\mid c) = \mathcal{N}(\mu_c, \Sigma_{0|c})$, then along each eigendirection $v_i$ of $\Sigma_{0|c}$ with eigenvalue $\lambda_i$,*
$$\mathrm{MMSE}_i(t) := v_i^\top \Sigma_{\mathrm{post}}(t)\, v_i = \frac{\lambda_i}{1 + \lambda_i\,\mathrm{SNR}(t)}. \tag{$\star$}$$

*Proof.* Monotonicity follows from the I-MMSE relation (Guo, Shamai, Verdú 2005, Thm. 1), which gives $\frac{d}{d\,\mathrm{SNR}}\mathrm{MMSE}(\mathrm{SNR}) = -\mathbb{E}\|\mathrm{MMSE}\text{ gradient}\|^2 \le 0$. The closed form $(\star)$ is the Wiener-filter posterior variance for the Gaussian channel $z_t = \alpha_t z_0 + \sigma_t\varepsilon$; see, e.g., Kay (1993, §11.4). $\square$

**Lemma 4** (Information-theoretic rate). *Under (A1, A4),*
$$\frac{d}{d\,\mathrm{SNR}}\,I(z_0;\,z_t\mid c) \;=\; \tfrac{1}{2}\,\mathrm{MMSE}(\mathrm{SNR}),$$
*so the cumulative MMSE decay equals (twice) the accumulated mutual information between $z_0$ and $z_t$.*

*Proof.* I-MMSE identity (Guo–Shamai–Verdú 2005). $\square$

### 2.3 Main theorem

**Theorem 1** (Early predictability of final quality). *Assume (A1)–(A4). For every $t\in[0,T]$,*
$$\mathbb{E}\bigl[(s_t - s_0^\star)^2\bigr] \;\le\; L^2 \cdot \mathrm{tr}\,\Sigma_{\mathrm{post}}(t), \tag{1}$$
*and the right-hand side is monotonically non-increasing in $\mathrm{SNR}(t)$.*

*Proof.* By (A2, A3), $q\circ\mathcal{D}$ is $L$-Lipschitz, hence
$$|s_t - s_0^\star| = |q(\mathcal{D}(\hat z_{0|t})) - q(\mathcal{D}(z_0^\star))| \le L\,\|\hat z_{0|t} - z_0^\star\|_2.$$
Squaring and taking expectations:
$$\mathbb{E}[(s_t - s_0^\star)^2] \le L^2\,\mathbb{E}\|\hat z_{0|t} - z_0^\star\|_2^2 = L^2\,\mathrm{tr}\,\mathrm{Cov}(z_0 - \hat z_{0|t}\mid c) = L^2\,\mathrm{tr}\,\Sigma_{\mathrm{post}}(t),$$
where the last equality uses Lemma 1 (so $\hat z_{0|t}$ is the MMSE estimator and the residual has covariance $\Sigma_{\mathrm{post}}(t)$). Monotonicity in $\mathrm{SNR}(t)$ is Lemma 3. $\square$

**Theorem 2** (PAC license for early stopping). *Assume (A1)–(A4) and that the law of $z_t$ conditional on $c$ is absolutely continuous with a log-concave density (satisfied, e.g., in the Gaussian case). Then for every $\varepsilon > 0$,*
$$\Pr\bigl[\,|s_t - \mathbb{E}[s_0^\star\mid\mathcal{F}_t]| > \varepsilon\,\bigr] \;\le\; 2\exp\!\left(-\frac{\varepsilon^2}{2\,L^2\,\mathrm{tr}\,\Sigma_{\mathrm{post}}(t)}\right). \tag{2}$$
*Consequently, for every tolerance $\varepsilon>0$ and confidence $\delta\in(0,1)$, there exists $t^\star = t^\star(\varepsilon,\delta) < T$ such that $\Pr[|s_t - s_0^\star|>\varepsilon] < \delta$ for every $t \le t^\star$.*

*Proof.* Since $q\circ\mathcal{D}$ is $L$-Lipschitz, the random variable $s_t$ is an $L$-Lipschitz function of $z_t$. The Gaussian / log-concave concentration inequality (Borell–TIS; see Ledoux 2001, Prop. 1.8) gives $(2)$ with the MMSE term identified via Theorem 1. Monotonicity of the RHS in $\mathrm{SNR}(t)$ and $\mathrm{tr}\,\Sigma_{\mathrm{post}}(t)\to 0$ as $\mathrm{SNR}(t)\to\infty$ yield the existence of $t^\star$. $\square$

### 2.4 Spectral corollary: why semantics converge before details

**Corollary 1** (Semantic-detail separation). *In the Gaussian setting, decompose the quality scorer's effective input subspace into a "semantic" part $V_{\mathrm{sem}}$ (top-$k$ eigendirections of $\Sigma_{0|c}$, eigenvalues $\lambda_1\ge\cdots\ge\lambda_k$) and a "detail" part $V_{\mathrm{det}}$ (eigenvalues $\lambda_{k+1},\ldots$). Then*
$$\mathbb{E}[(s_t - s_0^\star)^2] \;\le\; L^2\Bigl[\underbrace{\sum_{i\le k} \tfrac{\lambda_i}{1+\lambda_i\mathrm{SNR}(t)}}_{\text{semantic error}} + \underbrace{\sum_{i>k} \tfrac{\lambda_i}{1+\lambda_i\mathrm{SNR}(t)}}_{\text{detail error}}\Bigr],$$
*and the semantic error drops below any $\eta>0$ as soon as $\mathrm{SNR}(t)\ge (1/\eta - 1)/\lambda_k$, whereas the detail error requires $\mathrm{SNR}(t)\ge (1/\eta - 1)/\lambda_{k+1}$ — a gap of factor $\lambda_k/\lambda_{k+1}$ in the required SNR.*

*Proof.* Plug $(\star)$ into Theorem 1 and split the sum by index. The threshold follows from solving $\lambda_i/(1+\lambda_i\mathrm{SNR}) = \eta$. $\square$

This recovers the empirical ordering in the paper's Fig. 10 (Overall Consistency — a semantic metric — converges well before Imaging Quality — a detail metric).

### 2.5 What is proved, what is assumed, what is empirical

| Statement | Status |
|---|---|
| $\hat z_{0|t}$ is the MMSE estimate of $z_0$ | **Proved** (Lemma 1, only needs A1) |
| $\{\hat z_{0|t}\}$ is a martingale | **Proved** (Lemma 2, only needs Tower property) |
| $\mathrm{tr}\,\Sigma_{\mathrm{post}}(t)\!\downarrow$ as $\mathrm{SNR}(t)\!\uparrow$ | **Proved** (Lemma 3, I-MMSE) |
| Closed-form $(\star)$ | **Proved under Gaussian $p(z_0\mid c)$**; a sharp upper bound otherwise |
| Inequality (1) in Theorem 1 | **Proved** given (A1)–(A4) |
| PAC bound (2) in Theorem 2 | **Proved** under Gaussian / log-concave $p_t(z_t\mid c)$ |
| Specific value $t^\star \approx T - 10$ | **Empirical** — depends on the spectrum of $\Sigma_{0|c}$ and the Lipschitz constants $L_q, L_\mathcal{D}$ |

In particular, Theorems 1 and 2 jointly establish the *mathematical license* for quality-gated early stopping: the observable $s_t$ tracks the final score $s_0^\star$ with an error whose variance contracts monotonically and concentrates sub-Gaussianly. The *operational threshold* at which $s_t$ becomes reliable enough to act on must be calibrated empirically — that is the role of §3.

### 2.6 Extensions and what the proof does not cover

- **Approximate scores.** Replacing the Bayes-optimal $\hat z_{0|t}$ with a learned $\hat z^\theta_{0|t}$ that satisfies $\mathbb{E}\|\hat z^\theta_{0|t} - \hat z_{0|t}\|^2 \le \eta_\theta^2$ adds $L\eta_\theta$ to the RHS of (1) by the triangle inequality. This recovers the Chen–Chewi–Li–Lee–Salim–Zhang (ICML 2023) regime of $\varepsilon$-accurate score networks.
- **Non-Gaussian data.** Lemmas 1, 2, 4 hold without Gaussianity. The closed form $(\star)$ and hence Corollary 1 do not; a weaker version holds via the log-Sobolev / Brascamp–Lieb inequality for strongly log-concave $p(z_0\mid c)$.
- **Lipschitz constants.** $L_q, L_\mathcal{D}$ are assumed existent but not computed. For ViCLIP and the CogVideoX decoder, empirical Lipschitz estimates (e.g., via spectral norm power iteration) are a prerequisite for turning (1) and (2) into quantitative stopping rules.
- **Flow-matching / rectified flow.** The argument transfers with $\mathrm{SNR}(t)$ replaced by the signal-to-noise of the Gaussian path induced by the chosen schedule; Tweedie is still valid because the marginal remains Gaussian-conditioned.

These extensions are straightforward but mechanical; they are noted here to delimit exactly what is and is not contained in Theorems 1–2.

---

## 3. A Predictive Model for Prompt → Convergence Speed

Goal: given a prompt $p$, predict the alignment-score trajectory $\{s_t(p)\}$ or its summary statistics, so the detector becomes **prior-driven** rather than a purely on-line sliding window.

### Modeling target: parametric convergence curve

Fit each sample's trajectory $\{s_k\}$ with a three-parameter family

$$s_t(p) \;=\; s_\infty(p) - \bigl[s_\infty(p) - s_0(p)\bigr] \cdot \exp\!\bigl(-\kappa(p)\cdot (T-t)\bigr) + \eta_t$$

Each parameter is interpretable: $s_\infty$ is the expected final score (failure iff $s_\infty<\tau$), $\kappa$ is the convergence rate (the "dog vs. shark" gap in Fig. S2), $s_0$ is the initial level.

### Prompt features (three parallel channels: interpretable + predictive)

1. **Semantic-density features (interpretable):** entity count, action count, modifier count, spatial-relation tokens ("left of", "behind"), style tokens, negations — all extractable in one spaCy pass. These directly predict $\kappa$: compositional / relational prompts tend to have smaller $\kappa$ (slower convergence).
2. **Distributional rarity:** distance from CLIP text embedding to training/reference-corpus centroid, k-NN distance, per-token log-frequency. Rare prompts are both harder and slower.
3. **Raw CLIP text embedding:** 512-dim, as a fallback.

### Architecture

Do not use an end-to-end large Transformer — a few thousand trajectories cannot support it, and interpretability suffers. Recommended:

- **Head:** 2-layer MLP, input = [CLIP emb ⊕ interpretable features], output = mean and log-variance of $(s_\infty, \kappa, s_0)$ (Gaussian likelihood).
- **Loss:** given historical trajectories $\{s_k^{(i)}\}$, maximize the log-likelihood of the parametric curve (heteroscedastic Gaussian NLL, Kendall & Gal 2017 style).
- **Uncertainty:** Deep Ensemble (5 seeds) or MC Dropout. The uncertainty matters — the threshold $\tau$ should adapt to the predictive variance for each prompt.
- **Scale:** 0.1–1M parameters. Target: < 1 ms at inference so it runs before generation starts.

### Training data

Reuse the paper's own pipeline: the 1,800-prompt × per-step-score dataset is already implied by Fig. 10 / Fig. 11. Prompts from VBench + Panda70M are diverse enough. Train per-model (CogVideoX and Wan2.1 have different schedules).

### Plugging back into the detector (the real payoff)

The paper's dynamic detector is hand-crafted sliding-window logic. With this predictor, it upgrades to **Bayesian early-stopping**:

- **Prior:** $p(s_\infty, \kappa \mid \text{prompt})$ from the predictor.
- **Observations:** on-line scores $\{s_1,\ldots,s_k\}$ up to step $k$.
- **Posterior:** standard Kalman / Gaussian update → $p(s_\infty \mid \text{prompt}, s_{1:k})$.
- **Decision:** reject once $\Pr(s_\infty < \tau \mid \cdot) > 1-\alpha$, accept once $<\alpha$, otherwise keep observing.

This promotes Fig. S2's "sample-adaptive" idea from a heuristic to a Bayes-optimal stopping rule (regression version of Wald's SPRT), and under Gaussian assumptions inherits Chernoff's (1959) **minimum-expected-stopping-time** guarantee for free — a nontrivial theoretical bonus for sample-adaptive detection.

### Trade-offs

- Main risk: trajectory noise $\eta_t$ is not exactly Gaussian, and the parametric curve may fail to capture multi-modal behavior (e.g., trajectory jumps after Trial 1 / Trial 2 intervention). Train only on non-intervened baseline trajectories.
- Not worth doing: predicting the full trajectory as a sequence — SNR is too low; compress to three parameters and fit.

---

## 4. Implementation Roadmap

1. **Theory sanity check:** run a linear-Gaussian toy experiment to verify that the slope of the coarse-to-fine MMSE curve (from Layer A / §2-(iv)) matches the shape of Fig. 10 quantitatively.
2. **Minimal prompt predictor:** hook a small prompt→$(s_\infty, \kappa)$ head into the paper's codebase, replacing the fixed-window dynamic detector with the Bayesian early-stopping variant from §3.

Each step takes ~1–2 weeks and together they close both the theoretical and the modeling gaps identified in the paper.
