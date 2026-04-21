# Klima Protocol — Fee-Based vs. Principal Model

A carbon-retirement intermediary buys kVCM tokens and burns them on
behalf of clients. Two pricing regimes sit at the root of the
business: the **fee** book marks up the token cost pass-through (no
inventory, no exposure); the **principal** book quotes a fixed USD
price up front. We derive closed-form moments for each, decompose
principal exposure into a zero-capital **operating book** and a
balance-sheet **treasury**, and isolate three levers on the retained
tail: pre-purchased inventory $(k, C_{\mathrm{basis}})$, syndication
fraction $\beta$, and barrier-triggered switching at threshold $h$.
The numerical companion exercises every identity below.

## Setup and notation {#setup-and-notation}

The intermediary retires $\lambda$ tonnes per unit time over $[0, T]$.
Each tonne requires $P$ kVCM, sourced at spot $S_t$. Demand is
deterministic; $S_t$ is the only random driver, modelled as GBM,

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t,
\qquad S_0 = \pi_0 / P,
$$

with $\pi_t := P S_t$ the carbon price per tonne. Write $a \wedge b :=
\min(a, b)$.

*Price process.*

| Symbol | Meaning |
| --- | --- |
| $S_t$ | kVCM spot (USD / kVCM) |
| $S_0$ | spot at $t = 0$; $S_0 = \pi_0 / P$ |
| $\pi_0$ | initial carbon price per tonne (USD) |
| $\pi_t = P S_t$ | carbon price per tonne |
| $\mu, \sigma$ | GBM drift and volatility (annualised) |
| $W_t$ | driving Brownian motion |
| $\mathcal{F}_t$ | natural filtration |

*Contract and demand.*

| Symbol | Meaning |
| --- | --- |
| $P$ | kVCM per tonne (protocol constant) |
| $\lambda$ | retirement flow, tonnes / unit time |
| $T$ | horizon |
| $N = \lambda T$ | total tonnes retired |
| $f$ | fee rate |
| $Q$ | fixed USD quote per tonne (principal) |

*Treasury.*

| Symbol | Meaning |
| --- | --- |
| $k$ | treasury token notional (kVCM) |
| $C_{\mathrm{basis}}$ | sunk USD basis |
| $\alpha = \min(1, k / (N P))$ | coverage fraction |
| $\tau_{\mathrm{cov}} = \min(T, k / (P \lambda))$ | inventory-exhaustion time |

*Syndication.*

| Symbol | Meaning |
| --- | --- |
| $\beta \in [0, 1]$ | ceded fraction |
| $\theta \ge 0$ | counterparty risk load |
| $\pi_{\mathrm{syn}}$ | up-front premium (USD) |

*Switching.*

| Symbol | Meaning |
| --- | --- |
| $h \ge 1$ | threshold multiple |
| $H := h S_0$ | absolute threshold |
| $\mathbf{1}^{\mathrm{fee}}_t := \mathbf{1}\{S_t \ge H\}$ | fee-mode indicator |
| $\tau := \inf\{t : S_t \ge H\} \wedge T$ | first-passage time |
| $f_{\mathrm{post}}$ | fee rate in fee mode |

*Jumps (compensated Merton overlay; §[Jumps](#compensated-merton-jump-diffusion)).*

| Symbol | Meaning |
| --- | --- |
| $\lambda_J$ | Poisson intensity (/yr) |
| $\mu_J$ | mean log-jump |
| $\sigma_J$ | log-jump SD |
| $\kappa = e^{\mu_J + \sigma_J^2 / 2} - 1$ | Merton compensation |

P&L figures are totals over $[0, T]$ in USD; divide by $N$ for a
per-tonne reading.

### Risk metrics {#risk-metrics}

$$
\mathrm{VaR}_p[\Pi] := -\inf\{ x : \mathbb{P}[\Pi \le x] \ge 1 - p \},
\qquad
\mathrm{CVaR}_p[\Pi] := \mathbb{E}\!\left[ -\Pi \mid \Pi \le -\mathrm{VaR}_p[\Pi] \right],
$$

$$
\mathrm{Sharpe}[\Pi] := \mathbb{E}[\Pi] / \mathrm{SD}[\Pi]
\qquad \text{(horizon-absolute),}
$$

$$
z := (\mathbb{E}_{\mathrm{mc}}[\Pi] - \mathbb{E}_{\mathrm{cf}}[\Pi]) / \mathrm{stderr}
\qquad (\lvert z \rvert \le 2 \text{ sampling; } \lvert z \rvert > 3 \text{ suspect}).
$$

MC means carry $\pm\mathrm{CI}_{95} = 1.96 \cdot \mathrm{SD} / \sqrt{n}$.
$\mathrm{Sharpe}[R_{\mathrm{fee}}]$ is $f$-invariant; on the fee book
$R_{\mathrm{fee}} \ge 0$, so its $\mathrm{VaR}_p, \mathrm{CVaR}_p$ read
as low-end revenue rather than loss.

### The shared kernel

Every operating book below reduces, up to sign and scale, to

$$
I_T := \int_0^T S_t \, dt,
$$

with Dufresne (2001) moments

$$
\mathbb{E}[I_T] = S_0 \cdot \frac{e^{\mu T} - 1}{\mu}
  \quad (\to S_0 T \text{ as } \mu \to 0),
$$

$$
\mathbb{E}[I_T^2] = \frac{2 S_0^2}{\mu + \sigma^2}
  \left[
    \frac{e^{(2\mu + \sigma^2) T} - 1}{2\mu + \sigma^2}
    - \frac{e^{\mu T} - 1}{\mu}
  \right],
  \qquad
  \mathrm{Var}[I_T] = \mathbb{E}[I_T^2] - \mathbb{E}[I_T]^2.
$$

$I_T$ is not log-normal, so tail quantiles require Monte Carlo.

## The fee operating book {#fee-book}

Quote $(1 + f) \pi_t$, remit $\pi_t$, keep $f \pi_t$ per tonne:

$$
R_{\mathrm{fee}}
  = \int_0^T f \pi_t \lambda \, dt
  = f P \lambda \cdot I_T,
$$

$$
\mathbb{E}[R_{\mathrm{fee}}] = f P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[R_{\mathrm{fee}}] = (f P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

$R_{\mathrm{fee}} \ge 0$ a.s.; variance is driven entirely by $\sigma$;
$\mathrm{Sharpe}[R_{\mathrm{fee}}]$ is invariant in $f$.

## The back-to-back operating book {#back-to-back-book}

Fix $Q$ at inception, source each tonne at spot:

$$
\Pi_{\mathrm{b2b}}
  = \int_0^T (Q - P S_t) \lambda \, dt
  = Q N - P \lambda \cdot I_T,
$$

$$
\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q N - P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[\Pi_{\mathrm{b2b}}] = (P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

Hence $\mathrm{Var}[\Pi_{\mathrm{b2b}}] / \mathrm{Var}[R_{\mathrm{fee}}]
= (P/f)^2$ exactly; one MC pass prices both. Upside is capped at
$Q N$, downside unbounded. Economically equivalent to shorting a
continuous strip of forwards on kVCM struck at $Q / P$.

**Delta.** By Itô,

$$
\frac{\partial \, \mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi_{\mathrm{b2b}}(t) \mid \mathcal{F}_t]}{\partial S_t}
  = -P \lambda \cdot \frac{e^{\mu(T - t)} - 1}{\mu}
  \approx -P \lambda (T - t).
$$

The fee book satisfies the same identity with sign $+$ and magnitude
$f P \lambda (T - t)$. The natural static hedge for b2b at time $t$ is
$P \lambda (T - t)$ tokens of spot kVCM — precisely the treasury
schedule at $k = N P$.

## The active treasury {#treasury-book}

The treasury opens at $t = 0$ with $k$ tokens at basis
$C_{\mathrm{basis}}$, feeds retirement at spot, marks any over-hedge
leftover at $S_T$. Set

$$
\tau_{\mathrm{cov}} := \min\!\left(T, \frac{k}{P \lambda}\right),
\qquad
k_{\mathrm{left}} := \max(0, k - N P).
$$

Terminal P&L:

$$
\Pi_{\mathrm{trea}}
  = P \lambda \int_0^{\tau_{\mathrm{cov}}} S_t \, dt
  + k_{\mathrm{left}} \cdot S_T
  - C_{\mathrm{basis}},
\qquad
\Pi_{\mathrm{trea}}(0) = k \cdot S_0 - C_{\mathrm{basis}}.
$$

**Under-/exactly hedged** ($k \le N P$):

$$
\mathbb{E}[\Pi_{\mathrm{trea}}]
  = P \lambda \cdot S_0 \cdot \frac{e^{\mu \tau_{\mathrm{cov}}} - 1}{\mu}
  - C_{\mathrm{basis}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{trea}}]
  = (P \lambda)^2 \cdot \mathrm{Var}\!\left[\int_0^{\tau_{\mathrm{cov}}} S_t \, dt\right].
$$

**Over-hedged** ($k > N P$):

$$
\mathbb{E}[\Pi_{\mathrm{trea}}]
  = P \lambda \cdot \mathbb{E}[I_T] + k_{\mathrm{left}} \cdot S_0 \, e^{\mu T} - C_{\mathrm{basis}},
$$

$$
\mathrm{Var}[\Pi_{\mathrm{trea}}]
  = (P \lambda)^2 \mathrm{Var}[I_T]
  + k_{\mathrm{left}}^2 \, \mathrm{Var}[S_T]
  + 2 P \lambda \, k_{\mathrm{left}} \, \mathrm{Cov}[I_T, S_T],
$$

with $\mathrm{Var}[S_T] = S_0^2 e^{2\mu T} (e^{\sigma^2 T} - 1)$ and

$$
\mathrm{Cov}[I_T, S_T]
  = S_0^2 e^{\mu T} T
    \left( \frac{e^{(\mu + \sigma^2) T} - 1}{(\mu + \sigma^2) T}
         - \frac{e^{\mu T} - 1}{\mu T} \right),
$$

from $\mathbb{E}[S_t S_T] = S_0^2 e^{\mu(t + T) + \sigma^2 t}$ for $t
\le T$, integrated over $[0, T]$. The opening MTM $k S_0 -
C_{\mathrm{basis}}$ translates without reshaping.

## Strategy composition = operating + treasury {#strategy-composition}

$$
\Pi_{\mathrm{desk}} = \Pi_{\mathrm{op}} + \Pi_{\mathrm{trea}}.
$$

| Strategy | Operating | Treasury $(k, C_{\mathrm{basis}})$ |
| --- | --- | --- |
| Fee-only | fee | $(0, 0)$ |
| B2b | b2b | $(0, 0)$ |
| Matched | b2b | $(N P, N P S_0)$ |
| Partial ($\alpha$) | b2b | $(\alpha N P, \alpha N P S_0)$ |
| Custom | b2b | $(k, C_{\mathrm{basis}})$ |
| Syndicated | retained | $(0, 0)$ |
| Syndicated-matched | retained | $(N P, N P S_0)$ |
| Switching | switching | $(0, 0)$ |
| Switching-matched | switching | $(N P, N P S_0)$ |

At $(k, C_{\mathrm{basis}}) = (N P, N P S_0)$ the kernel cancels
path-by-path:

$$
\Pi_{\mathrm{matched}}
  = (Q N - P \lambda I_T) + (P \lambda I_T - N P S_0)
  = N (Q - P S_0),
$$

deterministic. Asserted to machine precision in
`test/models.test.ts`.

### Partial-desk closed form

Let $J_\alpha := \int_{\alpha T}^T S_t \, dt$. Then

$$
\Pi_{\mathrm{partial}}
  = \Pi_{\mathrm{b2b}} + \Pi_{\mathrm{trea}}
  \bigl|_{(\alpha N P, \alpha N P S_0)}
  = Q N - \alpha N P S_0 - P \lambda \, J_\alpha.
$$

By the strong Markov property at $\alpha T$, $J_\alpha = S_{\alpha T}
\cdot Y$ with $Y := \int_0^{(1-\alpha) T} S'_s \, ds$ an independent
unit-start GBM integral, so

$$
\mathbb{E}[J_\alpha]
  = S_0 e^{\mu \alpha T}
    \cdot (1-\alpha) T \cdot \frac{e^{\mu (1-\alpha) T} - 1}{\mu (1-\alpha) T},
$$

$$
\mathrm{Var}[J_\alpha]
  = S_0^2 e^{(2 \mu + \sigma^2) \alpha T} \cdot \mathbb{E}[Y^2]
  - \bigl(S_0 e^{\mu \alpha T} \cdot \mathbb{E}[Y]\bigr)^2,
$$

with $\mathbb{E}[Y^2]$ given by the Dufresne identity at $S_0 = 1$,
horizon $(1-\alpha) T$. Hence

$$
\mathbb{E}[\Pi_{\mathrm{partial}}]
  = Q N - \alpha N P S_0 - P \lambda \, \mathbb{E}[J_\alpha],
\qquad
\mathrm{Var}[\Pi_{\mathrm{partial}}]
  = (P \lambda)^2 \, \mathrm{Var}[J_\alpha].
$$

Boundary cases: $\alpha = 0 \Rightarrow J_0 = I_T$ recovers b2b;
$\alpha = 1 \Rightarrow J_1 = 0$ recovers the matched identity.
Between them, variance decays with the uncovered-window length, not
with $(1-\alpha)^2$.

## The syndicated-on-b2b operating book {#syndicated-book}

Cede fraction $\beta$ of the b2b operating book against premium
$\pi_{\mathrm{syn}}$:

$$
\Pi_{\mathrm{ret}}
  = (1 - \beta) \Pi_{\mathrm{b2b}} + \pi_{\mathrm{syn}},
\qquad
\Pi_{\mathrm{ret}}(0) = \pi_{\mathrm{syn}}.
$$

$$
\mathbb{E}[\Pi_{\mathrm{ret}}]
  = (1 - \beta) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{ret}}]
  = (1 - \beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Cession is $\alpha$-free at the operating layer; $\alpha$ re-enters
at the desk composition.

**Pricing.** The per-unit risk load is

$$
\rho(\theta) =
\begin{cases}
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}]
  & \text{sharpe,} \\
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] \cdot \phi(\Phi^{-1}(0.95)) / 0.05
  & \text{cvar,}
\end{cases}
$$

with shape factor $\approx 2.063$. The loaded premium is

$$
\pi_{\mathrm{syn}}(\beta, \theta)
  = \beta \bigl(\mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta)\bigr).
$$

At $\theta = 0$ the premium is actuarially fair and
$\mathbb{E}[\Pi_{\mathrm{ret}}]$ is $\beta$-invariant; for $\theta >
0$ the intermediary pays expected P&L for tail relief. The `cvar`
mode is a Gaussian surrogate: $I_T$'s tail is heavier than Gaussian,
so MC remains authoritative. Tranched cessions $(\max(L - K, 0))$
break the Dufresne backbone and are out of scope.

## The switching operating book {#switching-book}

Treasury and syndication rescale the loss tail; switching truncates
it. Whenever $S_t$ is above $H = h S_0$ the book quotes at rate
$f_{\mathrm{post}}$, yielding non-negative fee revenue on that
sub-interval. The mode indicator tracks the spot symmetrically:

$$
M_t := \begin{cases}
  \text{fee} & S_t \ge H, \\
  \text{b2b} & S_t < H,
\end{cases}
\qquad
\mathbf{1}^{\mathrm{fee}}_t := \mathbf{1}\{S_t \ge H\}.
$$

Split the kernel and the occupation times,

$$
I_{\mathrm{b2b}} := \int_0^T (1 - \mathbf{1}^{\mathrm{fee}}_t) S_t \, dt,
\qquad
I_{\mathrm{fee}} := \int_0^T \mathbf{1}^{\mathrm{fee}}_t S_t \, dt,
\qquad
I_{\mathrm{b2b}} + I_{\mathrm{fee}} = I_T,
$$

$$
T_{\mathrm{b2b}} := \int_0^T (1 - \mathbf{1}^{\mathrm{fee}}_t) \, dt,
\qquad
T_{\mathrm{fee}} := T - T_{\mathrm{b2b}}.
$$

The switching book's P&L is

$$
\Pi_{\mathrm{sw}}
  = Q \lambda T_{\mathrm{b2b}}
  - P \lambda I_{\mathrm{b2b}}
  + f_{\mathrm{post}} P \lambda I_{\mathrm{fee}},
  \qquad \Pi_{\mathrm{sw}}(0) = 0.
$$

Boundary: $h \to \infty$ recovers $\Pi_{\mathrm{b2b}}$; $h \le 1$
starts in fee mode.

### Closed-form anchors

With $\nu := \mu - \tfrac12 \sigma^2$ and $\log S_t \sim \mathcal{N}(\log S_0 + \nu t, \sigma^2 t)$,

$$
\mathbb{P}[S_t \ge H]
  = \Phi\!\left( \frac{\nu t - \log h}{\sigma \sqrt{t}} \right).
$$

The lognormal partial-expectation identity at $(m, v^2) = (\nu t,
\sigma^2 t)$ yields

$$
\mathbb{E}[S_t \mathbf{1}\{S_t \ge H\}]
  = S_0 e^{\mu t}
    \cdot \Phi\!\left(
      \frac{\mu t + \tfrac12 \sigma^2 t - \log h}{\sigma \sqrt{t}}
    \right),
$$

so by Fubini

$$
\mathbb{E}[T_{\mathrm{fee}}]
  = \int_0^T \Phi\!\left( \frac{\nu t - \log h}{\sigma \sqrt{t}} \right) dt,
$$

$$
\mathbb{E}[I_{\mathrm{fee}}]
  = S_0 \int_0^T e^{\mu t} \Phi\!\left(
      \frac{\mu t + \tfrac12 \sigma^2 t - \log h}{\sigma \sqrt{t}}
    \right) dt.
$$

Both are Simpson-tractable (`expectedTimeAboveBarrier`,
`expectedIntegralAboveBarrier` in `src/core/moments.ts`).

The first-passage time $\tau$ is a path property. Its distribution
under pure GBM is the Harrison / Borodin-Salminen law:

$$
\mathbb{P}[\tau \le T]
  = \Phi\!\left(\frac{-\log h + \nu T}{\sigma \sqrt{T}}\right)
  + h^{2\nu / \sigma^2} \Phi\!\left(\frac{-\log h - \nu T}{\sigma \sqrt{T}}\right),
$$

with $\mathbb{E}[\tau \wedge T] = \int_0^T (1 - \mathbb{P}[\tau \le t]) \, dt$.
$T_{\mathrm{fee}} \neq T - \tau$ in general (post-$\tau$ re-entries
contribute), so $\mathbb{E}[\tau \wedge T]$ and $\mathbb{E}[T_{\mathrm{fee}}]$
are independent quantities. No closed-form density exists for
$\Pi_{\mathrm{sw}}$; tail quantiles require MC.

### Truncation vs rescaling

Partition by first-passage:

$$
\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]
  \le
  \mathbb{P}[\tau = T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau = T\}}[\Pi_{\mathrm{b2b}}]
  + \mathbb{P}[\tau < T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau < T\}}[\Pi_{\mathrm{sw}}].
$$

On $\{\tau = T\}$, $\Pi_{\mathrm{sw}} = \Pi_{\mathrm{b2b}}$; on $\{\tau
< T\}$ every fee-mode sub-interval contributes $\ge 0$. Tightening
$h \downarrow 1$ shrinks the first term and grows the second.
Re-entries temper the cap: paths that cross back below $H$ resume b2b
exposure. The simulator reports `CVaR95|no-switch` and
`CVaR95|switched` separately.

### Operator decision surface

Sweeping $h$ at fixed $(\mu, \sigma, f, f_{\mathrm{post}})$:
$\mathbb{E}[\Pi_{\mathrm{sw}}]$ and $\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]$
move monotonically in opposite directions. Optimal $h$ is a
control-problem formulation (out of scope).

## Direct comparison {#direct-comparison}

Operating books plus treasury, pure GBM:

| | Fee | B2b | Retained | Switching | Treasury $(k, C_{\mathrm{basis}})$ |
| --- | --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f P \lambda \mathbb{E}[I_T]$ | $Q N - P \lambda \mathbb{E}[I_T]$ | $(1 - \beta) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}}$ | MC; anchors on $\mathbb{E}[T_{\mathrm{fee}}]$, $\mathbb{E}[I_{\mathrm{fee}}]$ | $P \lambda \mathbb{E}[I_{\tau_{\mathrm{cov}}}] + k_{\mathrm{left}} S_0 e^{\mu T} - C_{\mathrm{basis}}$ |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \mathrm{Var}[I_T]$ | $(P \lambda)^2 \mathrm{Var}[I_T]$ | $(1 - \beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}]$ | MC | branch on $k_{\mathrm{left}}$ |
| kVCM exposure | long | short | short, $\times (1 - \beta)$ | short on $\{S_t < H\}$, long on $\{S_t \ge H\}$ | long on $[0, \tau_{\mathrm{cov}}]$, plus $S_T$ on leftover |
| Downside | $\ge 0$ | unbounded | $(1 - \beta) \times$ b2b | truncated on fee-mode intervals | $-C_{\mathrm{basis}}$ if $S \equiv 0$ |
| Capital | 0 | 0 | 0 | 0 | $C_{\mathrm{basis}}$ |
| Counterparty | none | none | $\beta \Pi_{\mathrm{b2b}}$ upside | as retained on the fee leg | none |

Desk totals are row-wise sums.

### Break-even quote

$\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$ solves to

$$
Q^* = (1 + f) P S_0 \cdot \frac{e^{\mu T} - 1}{\mu T},
$$

with $Q^* \to (1 + f) P S_0$ as $\mu \to 0$. $\mu > 0$ pushes $Q^*$
above that level; $\mu < 0$ below.

At $Q = Q^*$ the two books share $I_T$ and therefore variance, but
enter it with opposite signs: fee is bounded below, b2b carries a
left-skewed loss tail. This asymmetry — present even at
moment-equalised $Q$ — motivates the three dials.

## Jumps: compensated Merton overlay {#compensated-merton-jump-diffusion}

Replace pure GBM with

$$
\frac{dS_t}{S_{t-}}
  = (\mu - \lambda_J \kappa) \, dt
  + \sigma \, dW_t
  + (J - 1) \, dN_t,
$$

for $N_t$ Poisson($\lambda_J$) independent of $W_t$ and $J = e^Y$,
$Y \sim N(\mu_J, \sigma_J^2)$ i.i.d. The compensation $\kappa :=
\mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ removes the jump
mean from the drift.

**Means survive.** The compound-Poisson identity gives
$\mathbb{E}[S_t] = S_0 e^{\mu t}$ for every $(\lambda_J, \mu_J,
\sigma_J)$, so

$$
\mathbb{E}[I_T] = S_0 \cdot \frac{e^{\mu T} - 1}{\mu}
$$

and every mean-level identity above carries over: $\mathbb{E}[R_{\mathrm{fee}}]$,
$\mathbb{E}[\Pi_{\mathrm{b2b}}]$, $\mathbb{E}[\Pi_{\mathrm{trea}}]$,
$\mathbb{E}[\Pi_{\mathrm{ret}}]$, $Q^*$, and the matched-desk identity.

**Variances do not.** Jumps inflate $\mathrm{Var}[S_t]$, hence
$\mathrm{Var}[I_T]$ and every downstream tail metric. The simulator
reports the pure-GBM variance as a **GBM anchor** and leaves
jump-aware tails to MC. `test/jump-gbm.test.ts` verifies both
predictions.

## Limitations {#limitations-and-next-steps}

| Baseline | Lifted in |
| --- | --- |
| Deterministic demand | Simulator — compound-Poisson order flow |
| GBM dynamics | Simulator — compensated Merton (above); regime switching pending |
| No calibration | Simulator — kVCM proxy |
| No discounting / gas / slippage | Simulator — parameterised |
| Passive treasury | Simulator — active consumption schedule; dynamic delta hedge pending |
| Static syndication / switching | Simulator — quota-share cession; symmetric threshold; optimal-$h$ pending |
| No credit layer | Out of scope |

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Adv. Appl. Probab. 33(1), 223–241.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4.
- Harrison, J. M. (1985). *Brownian Motion and Stochastic Flow Systems.*
