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

**Price dynamics.** $S_t$ follows Geometric Brownian Motion under the physical measure:

$$
dS_t = \mu \cdot S_t \cdot dt + \sigma \cdot S_t \cdot dW_t, \quad S_0 \text{ given.}
$$

Because $P$ is a constant multiplier, $\pi_t = P \cdot S_t$ is itself GBM with the same $(\mu, \sigma)$.

**Demand.** Retirement flow is deterministic at rate $\lambda$ tonnes / unit time. The sole source of randomness in the baseline is the kVCM price.

**Ignored in the baseline.** Risk-free discounting, gas, on-chain slippage, order-flow stochasticity. Each is re-introduced in a later phase (see §6).

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

Mean and variance interpolate linearly in $\alpha$:

$$
\mathbb{E}[\Pi_\alpha] = (1 - \alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \cdot \Pi_{\mathrm{matched}},
$$

$$
\mathrm{Var}[\Pi_\alpha] = (1 - \alpha)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

$\alpha$ is the company's hedge ratio against spot. $\alpha = 1$ gives a deterministic P&L (fully hedged at inception); $\alpha = 0$ is the unhedged short-forward strip.

## 4. Risk quantification

For each model, the Phase B simulator should report:

| Metric | How |
| --- | --- |
| $\mathbb{E}[\Pi]$, $\mathrm{Var}[\Pi]$, $\mathrm{SD}[\Pi]$ | Closed form from §2–3 |
| $\mathrm{VaR}_{95}$, $\mathrm{VaR}_{99}$ | Monte Carlo empirical quantile of $-\Pi$ |
| $\mathrm{CVaR}_{95}$, $\mathrm{CVaR}_{99}$ | Monte Carlo tail mean of $-\Pi$ |
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

| | Fee-based | Principal 3a (matched) | Principal 3b (back-to-back) |
| --- | --- | --- | --- |
| $\mathbb{E}[\text{P\&L}]$ | $f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T]$ | $N \cdot (Q - P \cdot S_0)$ | $Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T]$ |
| $\mathrm{Var}[\text{P\&L}]$ | $(f P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | $0$ (terminal) | $(P \lambda)^2 \cdot \mathrm{Var}[I_T]$ |
| kVCM exposure | long | none (terminal), long (interim NAV) | short |
| Downside | bounded below by 0 | deterministic | unbounded |
| Capital requirement | none | $N \cdot P \cdot S_0$ | none |

### Break-even quote

The principal back-to-back model matches the fee-based model's *expected* revenue when

$$
Q^* = (1 + f) \cdot P \cdot \mathbb{E}[I_T] / T = (1 + f) \cdot P \cdot S_0 \cdot \frac{e^{\mu T} - 1}{\mu T}.
$$

As $\mu \to 0$, $Q^* \to (1 + f) \cdot P \cdot S_0$ — the fee-based time-zero quote. For $\mu > 0$ the principal model must quote *above* that to compensate for expected kVCM appreciation; for $\mu < 0$ it quotes below.

**Asymmetry observation.** Even at $Q = Q^*$, the two books have radically different risk profiles: the fee book has bounded positive-only revenue, while the back-to-back principal book has the same variance but a *left-skewed* loss tail. Matching means does not match distributions.

## 6. Limitations and next steps

| Baseline simplification | Removed in |
| --- | --- |
| Deterministic demand | Phase C — compound-Poisson order flow |
| GBM price dynamics | Phase C — Merton jump-diffusion and/or two-state regime switching |
| No historical calibration | Phase C — kVCM proxy (KLIMA, BCT, NCT) |
| No discounting, gas, or on-chain slippage | Phase B — parameterized |
| Static (or absent) hedging | Phase C — dynamic delta hedge with inventory; perp/futures hedge if available |
| No credit / counterparty layer | Not scoped |

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Advances in Applied Probability, 33(1), 223–241. — closed-form moments of $I_T$.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4. — simulation of path integrals of GBM.
