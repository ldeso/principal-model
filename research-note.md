# Klima Protocol — Fee-Based vs. Principal Model

A research note on revenue and risk quantification for a carbon-retirement intermediary operating against the Klima Protocol.

This is **Phase A** of a three-phase roadmap:

- Phase A (this note) — mathematical framework, closed-form moments under GBM, risk-metric menu.
- Phase B — TypeScript simulation reproducing the results below.
- Phase C — jump-diffusion / regime-switching extensions, Poisson demand, historical calibration against a kVCM proxy, hedging strategies.

## 1. Setup and notation

| Symbol | Meaning |
| --- | --- |
| $S_t$ | kVCM/USD spot price at time $t$ |
| $P$ | Protocol price (constant) in kVCM per tonne |
| $\pi_t := P \cdot S_t$ | USD protocol price per tonne |
| $\lambda$ | Deterministic retirement flow, tonnes / unit time |
| $T$ | Horizon |
| $N := \lambda \cdot T$ | Total tonnes retired over $[0, T]$ |
| $f$ | Fee rate (e.g. 0.05) |
| $Q$ | Fixed USD quote per tonne in the principal model |
| $\alpha \in [0, 1]$ | Fraction of inventory pre-purchased (principal, generalized) |
| $\beta \in [0, 1]$ | Fraction of the $(1-\alpha)$ stochastic leg ceded to a third-party counterparty (§3d) |
| $\pi$ | Up-front premium received for ceding $\beta \cdot (1-\alpha)$ of the book |
| $\theta \geq 0$ | Counterparty risk-load multiplier (§3d) |

**Price dynamics.** $S_t$ follows Geometric Brownian Motion under the physical measure:

$$
dS_t = \mu \cdot S_t \cdot dt + \sigma \cdot S_t \cdot dW_t, \quad S_0 \text{ given.}
$$

Because $P$ is a constant multiplier, $\pi_t = P \cdot S_t$ is itself GBM with the same $(\mu, \sigma)$.

**Demand.** Retirement flow is deterministic at rate $\lambda$ tonnes / unit time. The sole source of randomness in the baseline is the kVCM price.

**Ignored in the baseline.** Risk-free discounting, gas, on-chain slippage, order-flow stochasticity. Each is re-introduced in a later phase (see §7).

### The central stochastic object

Both models' P&L are linear functionals of the **integral of the GBM**:

$$
I_T := \int_0^T S_t\,dt.
$$

Its first two moments are known in closed form (Dufresne, 2001):

$$
\mathbb{E}[I_T] = S_0 \cdot \frac{e^{\mu T} - 1}{\mu}, \quad \mu \neq 0 \quad (\to S_0 \cdot T \text{ as } \mu \to 0)
$$

$$
\mathbb{E}[I_T^2] = \frac{2 S_0^2}{\mu + \sigma^2} \left[ \frac{e^{(2\mu + \sigma^2) T} - 1}{2\mu + \sigma^2} - \frac{e^{\mu T} - 1}{\mu} \right],
$$

$$
\mathrm{Var}[I_T] = \mathbb{E}[I_T^2] - \mathbb{E}[I_T]^2.
$$

The distribution of $I_T$ is not log-normal, so tail metrics (VaR, CVaR) are obtained by Monte Carlo; the moments above serve as closed-form anchors and Monte Carlo sanity checks.

## 2. Fee-based model

The company quotes clients $(1 + f) \cdot \pi_t$ and remits $\pi_t$ to the protocol, keeping $f \cdot \pi_t$ per tonne.

**Total revenue over $[0, T]$:**

$$
R_{\mathrm{fee}} = \int_0^T f \cdot \pi_t \cdot \lambda\,dt = f \cdot P \cdot \lambda \cdot I_T.
$$

**Moments:**

$$
\mathbb{E}[R_{\mathrm{fee}}] = f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T].
$$

$$
\mathrm{Var}[R_{\mathrm{fee}}] = (f \cdot P \cdot \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

**Properties.**

- $R_{\mathrm{fee}} \geq 0$ almost surely.
- Top-line volatility is fully driven by $\sigma$ — the company has no balance-sheet exposure and holds no inventory.
- Scales linearly in $f$, so the risk-adjusted return per unit fee is invariant in $f$.

## 3. Principal model

The company sets a fixed USD quote $Q$ at $t = 0$. Three variants of inventory sourcing, in increasing order of risk:

### 3a. Fully matched pre-purchase ($\alpha = 1$)

At $t = 0$ buy exactly $N \cdot P$ kVCM at spot $S_0$; cost $C = N \cdot P \cdot S_0$. Burn against deterministic demand over $[0, T]$.

**Terminal P&L:**

$$
\Pi_{\mathrm{matched}} = N \cdot (Q - P \cdot S_0),
$$

which is **deterministic** — no terminal kVCM risk.

Risk still exists *interim*. The mark-to-market inventory value at time $t$ is

$$
V_t = (N \cdot P) \cdot (1 - t/T) \cdot S_t,
$$

which is a scaled GBM decayed by a deterministic burn schedule. Solvency, margin-call, or accounting-covenant concerns live in the distribution of $\max_{t \leq T} (V_0 - V_t)$ (max drawdown). This is tracked by Monte Carlo in Phase B.

### 3b. Back-to-back acquisition ($\alpha = 0$)

The company quotes $Q$ at $t = 0$ but buys kVCM at spot for each retirement. Per-tonne realized P&L is $Q - P \cdot S_t$.

**Total P&L:**

$$
\Pi_{\mathrm{b2b}} = \int_0^T (Q - P \cdot S_t) \cdot \lambda\,dt = Q \cdot N - P \cdot \lambda \cdot I_T.
$$

**Moments:**

$$
\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T].
$$

$$
\mathrm{Var}[\Pi_{\mathrm{b2b}}] = (P \cdot \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

Note that $\mathrm{Var}[\Pi_{\mathrm{b2b}}]$ coincides with $\mathrm{Var}[R_{\mathrm{fee}}]$ up to the rescaling factor $(P / f)^2 \cdot f^2 = P^2$ — i.e. the two models share **the same random kernel** $I_T$. Phase B can therefore reuse a single Monte Carlo simulation of $I_T$ and rescale.

**Payoff shape.** $\Pi_{\mathrm{b2b}}$ is linearly *decreasing* in $I_T$: upside is capped at $Q \cdot N$ (reached as $S_t \to 0$), downside is unbounded if kVCM rallies. Equivalent to shorting a continuous strip of forwards on kVCM struck at $Q / P$.

### 3c. Partial pre-purchase ($\alpha \in [0, 1]$)

Buy $\alpha \cdot N \cdot P$ kVCM at $S_0$, source the rest back-to-back. Then

$$
\begin{aligned}
\Pi_\alpha &= \alpha \cdot N \cdot (Q - P \cdot S_0) + (1 - \alpha) \cdot (Q \cdot N - P \cdot \lambda \cdot I_T) \\
&= Q \cdot N - P \cdot \lambda \cdot \left[ \alpha \cdot S_0 \cdot T + (1 - \alpha) \cdot I_T \right].
\end{aligned}
$$

The mean interpolates linearly in $\alpha$; the variance scales as $(1-\alpha)^2$:

$$
\mathbb{E}[\Pi_\alpha] = (1 - \alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \cdot \Pi_{\mathrm{matched}},
$$

$$
\mathrm{Var}[\Pi_\alpha] = (1 - \alpha)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

$\alpha$ is the company's hedge ratio against spot. $\alpha = 1$ gives a deterministic P&L (fully hedged at inception); $\alpha = 0$ is the unhedged short-forward strip.

### 3d. Quota-share syndication ($\beta \in [0, 1]$)

§3c is an **internal** hedge — the intermediary trades capital ($\alpha \cdot N \cdot P \cdot S_0$ at inception) for variance reduction on its own balance sheet. It does not address the §5 asymmetry observation that follows: at $Q = Q^*$ the principal book inherits the shared $I_T$ kernel with a *minus* sign, so mean preservation buys a left-skewed loss tail. Raising $\alpha$ contracts that tail, but every unit of contraction costs a unit of capital tied up in kVCM at $t = 0$.

Quota-share syndication is the complementary primitive. At $t = 0$ the intermediary sells a share $\beta \in [0, 1]$ of the $(1 - \alpha)$ stochastic leg to an external counterparty in exchange for an up-front premium $\pi$. No capital is tied up: the counterparty simply receives $\beta \cdot (1 - \alpha) \cdot \Pi_{\mathrm{b2b}}$ at $T$ against the premium paid at $0$. The matched slice has zero variance, so syndicating it is vacuous and $\beta$ touches the stochastic leg only.

**Retained P&L.** Combining §3a and §3b net of the ceded fraction,

$$
\Pi_{\mathrm{ret}} = \alpha \cdot N \cdot (Q - P \cdot S_0) + (1 - \alpha)(1 - \beta) \cdot \bigl( Q \cdot N - P \cdot \lambda \cdot I_T \bigr) + \pi.
$$

This is still **linear in $I_T$**, so the Dufresne-moment backbone carries over unchanged:

$$
\mathbb{E}[\Pi_{\mathrm{ret}}] = \alpha \cdot \Pi_{\mathrm{matched}} + (1 - \alpha)(1 - \beta) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi,
$$

$$
\mathrm{Var}[\Pi_{\mathrm{ret}}] = (1-\alpha)^2 (1-\beta)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

The variance collapses as the *product* $(1-\alpha)^2 (1-\beta)^2$ — $\alpha$ and $\beta$ are multiplicatively equivalent on the variance scale, but not on the capital or counterparty scale.

**Premium.** Write the per-unit-cession risk load as

$$
\rho(\theta) = \begin{cases} \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] & \text{sharpe mode}, \\ \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] \cdot \phi\!\bigl(\Phi^{-1}(0.95)\bigr)/0.05 & \text{cvar mode}, \end{cases}
$$

where $\phi, \Phi$ are the standard-normal density and CDF (the Gaussian shape factor evaluates to $\approx 2.063$). Then

$$
\pi(\alpha, \beta, \theta) = \beta \cdot (1 - \alpha) \cdot \bigl( \mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta) \bigr).
$$

Both modes keep $\pi$ a closed-form scalar, so the retained-variance identity above is **exact** under Monte Carlo (no sample-moment dependence). $\theta = 0$ yields the actuarially fair premium $\pi_{\mathrm{fair}} = \beta(1-\alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}]$, at which $\mathbb{E}[\Pi_{\mathrm{ret}}] = \mathbb{E}[\Pi_\alpha]$ independently of $\beta$ (no free lunch: mean preservation across cession levels). $\theta > 0$ buys the counterparty a risk premium — the intermediary sacrifices expected P&L in exchange for tail contraction.

The CVaR mode is a **Gaussian-approximate** surrogate — $I_T$ is not log-normal and its true tail is heavier than Gaussian, so Monte Carlo is authoritative for the retained book's tail metrics (CVaR₉₅, skew, $\mathbb{P}[\Pi_{\mathrm{ret}} < 0]$). The Gaussian-CVaR factor lets the sharpe and cvar modes be compared on a common SD unit.

**$Q^*$ invariance.** $Q^* = (1 + f) \cdot P \cdot S_0 \cdot (e^{\mu T} - 1)/(\mu T)$ is defined by $\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$, which predates the cession. Hence $Q^\ast$ is invariant in $\beta$ — the break-even quote is unaffected by how the principal book is syndicated.

**Capital vs. counterparty exposure.** $\alpha$ and $\beta$ are **orthogonal** along two balance-sheet dimensions: $\alpha$ consumes capital $\alpha N P S_0$ at $t = 0$ and eliminates spot beta on its slice; $\beta$ consumes no capital but exposes the intermediary to counterparty credit risk (scoped out here, consistent with §7). The two-dimensional $(\alpha, \beta)$ surface Pareto-dominates either axis alone for any counterparty who prices $\theta < \theta_{\mathrm{max}}$.

**Tranching is out of scope.** A first-loss / senior split would price the ceded leg as $\max(L - K, 0)$ — non-linear in $I_T$, breaking the Dufresne-moment backbone. It belongs in a separate MC-only note and is not part of this extension.

## 4. Risk quantification

For each model, the Phase B simulator should report:

| Metric | How |
| --- | --- |
| $\mathbb{E}[\Pi]$, $\mathrm{Var}[\Pi]$, $\mathrm{SD}[\Pi]$ | Closed form from §2–3 |
| VaR<sub>95</sub>, VaR<sub>99</sub> | Monte Carlo empirical quantile of $-\Pi$ |
| CVaR<sub>95</sub>, CVaR<sub>99</sub> | Monte Carlo tail mean of $-\Pi$ |
| $\mathbb{P}[\Pi < 0]$ | Monte Carlo |
| Sharpe-like $= \mathbb{E}[\Pi] / \mathrm{SD}[\Pi]$ | Closed form |
| Max NAV drawdown (principal 3a only) | Monte Carlo on $V_t$ path |

### Itô dynamics and delta

Applying the Itô product rule to the cumulative P&L process, the instantaneous sensitivity of *remaining* P&L to the spot $S_t$ is:

$$
\text{Fee-based:} \quad \frac{\partial\,\mathbb{E}[R_{\mathrm{fee}} - R(t) \mid \mathcal{F}_t]}{\partial S_t} = f \cdot P \cdot \lambda \cdot \frac{e^{\mu(T-t)} - 1}{\mu} \approx f \cdot P \cdot \lambda \cdot (T - t) \text{ for } \mu T \text{ small.}
$$

$$
\text{Principal 3b:} \quad \frac{\partial\,\mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi(t) \mid \mathcal{F}_t]}{\partial S_t} = -P \cdot \lambda \cdot \frac{e^{\mu(T-t)} - 1}{\mu} \approx -P \cdot \lambda \cdot (T - t).
$$

Signs are opposite: the fee book is **long** kVCM beta; the back-to-back principal book is **short** kVCM beta. The matched principal book has zero delta (fully pre-hedged by physical inventory).

This is the handle for Phase C hedging: the natural static hedge for the principal back-to-back book is to hold $(P \cdot \lambda) \cdot (T - t)$ tokens of spot kVCM at each time $t$ — which is exactly the matched-pre-purchase strategy (§3a) amortized to the remaining horizon.

## 5. Direct comparison

| | Fee-based | Principal 3a (matched) | Principal 3b (back-to-back) | Principal 3d (syndicated) |
| --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T]$ | $N \cdot (Q - P \cdot S_0)$ | $Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T]$ | $\mathbb{E}[\Pi_\alpha] - \beta(1-\alpha)\rho(\theta)$ |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | $0$ (terminal) | $(P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | $(1-\alpha)^2(1-\beta)^2 (P \lambda)^2 \cdot \mathrm{Var}[I_T]$ |
| kVCM exposure | long | none (terminal), long (interim NAV) | short | short, scaled by $(1-\beta)$ |
| Downside | bounded below by 0 | deterministic | unbounded | unbounded, scaled by $(1-\beta)$ |
| Capital requirement | none | $N \cdot P \cdot S_0$ | none | $\alpha \cdot N \cdot P \cdot S_0$ |
| Counterparty exposure | none | none | none | $\beta \cdot (1-\alpha) \cdot P \cdot \lambda \cdot I_T$ upside (if counterparty defaults) |

### Break-even quote

The principal back-to-back model matches the fee-based model's *expected* revenue when

$$
Q^* = (1 + f) \cdot P \cdot \mathbb{E}[I_T] / T = (1 + f) \cdot P \cdot S_0 \cdot \frac{e^{\mu T} - 1}{\mu T}.
$$

As $\mu \to 0$, $Q^* \to (1 + f) \cdot P \cdot S_0$ — the fee-based time-zero quote. For $\mu > 0$ the principal model must quote *above* that to compensate for expected kVCM appreciation; for $\mu < 0$ it quotes below.

**Asymmetry observation.** Matching means does not match distributions. At $Q = Q^*$ the two books share the kernel $I_T$ and therefore share variance; but the fee book enters $I_T$ with a plus sign (bounded below by $0$) while the back-to-back book enters it with a minus sign (left-skewed loss tail). The principal book trades mean preservation for a downside tail the fee book does not have.

## 6. Compensated Merton jump-diffusion

The first Phase C relaxation of the baseline replaces pure GBM with Merton's
jump-diffusion. The spot process becomes

$$
\frac{dS_t}{S_{t-}} = (\mu - \lambda_J \kappa)\,dt + \sigma\,dW_t + (J - 1)\,dN_t,
$$

where $N_t$ is a Poisson process with intensity $\lambda_J$ independent of
$W_t$, and at each arrival the price multiplies by $J = e^Y$ with
$Y \sim N(\mu_J, \sigma_J^2)$ i.i.d. The drift offset
$\kappa := \mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ is the **Merton
compensation** — it subtracts the jump component's mean effect so the pure-$W$
part carries the economic drift $\mu$.

### Means are invariant under compensation

The exact solution is

$$
S_t = S_0 \exp\!\Big(\big(\mu - \tfrac12 \sigma^2 - \lambda_J \kappa\big) t
      + \sigma W_t + \sum_{k = 1}^{N_t} Y_k\Big).
$$

Taking expectations and using independence of $W$, $N$, and $\{Y_k\}$:

$$
\mathbb{E}[S_t]
  = S_0 \, e^{(\mu - \lambda_J \kappa) t}
    \, \mathbb{E}\!\left[e^{\sigma W_t}\right]
    \, \mathbb{E}\!\left[\prod_{k=1}^{N_t} e^{Y_k}\right]
  = S_0 \, e^{(\mu - \lambda_J \kappa) t} \cdot e^{\sigma^2 t/2}
    \cdot e^{\lambda_J \kappa t}
  = S_0 \, e^{\mu t},
$$

using the generating-function identity
$\mathbb{E}\!\left[\prod_{k=1}^{N_t} e^{Y_k}\right]
  = e^{\lambda_J t (\mathbb{E}[e^Y] - 1)}
  = e^{\lambda_J \kappa t}$
for compound-Poisson exponentials. So $\mathbb{E}[S_t] = S_0 \cdot e^{\mu t}$ is
identical to the pure-GBM value, for every $(\lambda_J, \mu_J, \sigma_J)$.

Integrating over $[0, T]$ and swapping Fubini:

$$
\mathbb{E}[I_T] = \int_0^T \mathbb{E}[S_t]\,dt = S_0 \cdot \frac{e^{\mu T} - 1}{\mu},
$$

i.e. **Dufresne's GBM first-moment formula carries over unchanged**. Consequently
every mean-level quantity that depended on $I_T$ linearly does too:

- $\mathbb{E}[R_{\mathrm{fee}}] = f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T]$,
- $\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T]$,
- $\mathbb{E}[\Pi_\alpha] = (1-\alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \cdot \Pi_{\mathrm{matched}}$,
- $Q^* = (1+f) \cdot P \cdot S_0 \cdot (e^{\mu T} - 1)/(\mu T)$.

The matched book's §3a P&L is already pathwise-deterministic, so jumps don't
touch it either.

### Variance is not

Jumps do inflate $\mathrm{Var}[S_t]$ (and therefore $\mathrm{Var}[I_T]$ and every
downstream variance / tail metric). A closed form exists but is more involved;
Phase B reports the pure-GBM $\mathrm{Var}[I_T]$ as a **GBM anchor** and leaves
the true jump-aware tail metrics to Monte Carlo (the tests in
`test/jump-gbm.test.ts` confirm that the GBM closed-form means still match the
jump-MC means within CI, and that the empirical variance strictly exceeds the
GBM anchor).

### Itô deltas are unchanged in expectation

The jump contribution to $\mathbb{E}[S_t]$ is zero, so the §4 delta expressions and the §3a static hedge $(P \cdot \lambda) \cdot (T - t)$ carry over verbatim under compensated Merton — with fatter realised hedging error.

## 7. Limitations and next steps

This table is the authoritative scope statement for the programme; the landing page links here rather than duplicating it.

| Baseline simplification | Removed in |
| --- | --- |
| Deterministic demand | Phase C — compound-Poisson order flow |
| GBM price dynamics | Phase C — Merton jump-diffusion implemented (§6); two-state regime switching pending |
| No historical calibration | Phase C — kVCM proxy (KLIMA, BCT, NCT) |
| No discounting, gas, or on-chain slippage | Phase C — parameterized |
| Static (or absent) hedging | Phase C — dynamic delta hedge with inventory; perp/futures hedge if available; quota-share syndication is now in §3d |
| No credit / counterparty layer | Not scoped (§3d treats syndication as default-free; tranching remains out of scope — non-linear in $I_T$, would break the closed-form backbone) |

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Advances in Applied Probability, 33(1), 223–241. — closed-form moments of $I_T$.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4. — simulation of path integrals of GBM.
