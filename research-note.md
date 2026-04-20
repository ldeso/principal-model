# Klima Protocol — Fee-Based vs. Principal Model

A carbon-retirement intermediary buys kVCM tokens on the open market
and burns them to retire carbon credits on behalf of clients. Two
pricing choices sit at the root of the business. The **fee** book
passes the token cost through and adds a markup — no inventory, no
balance-sheet exposure. The **principal** book quotes a fixed USD
price up front. This note derives, in closed form, what each choice
delivers in expected revenue and tail risk, splits the principal
exposure into a zero-capital **operating book** and a balance-sheet
**treasury**, and identifies three dials by which the retained tail
can be reshaped: pre-buying inventory (a treasury notional $k$ with
basis $C_{\mathrm{basis}}$), syndicating the operating book
($\beta$), and barrier-triggered switching ($h$). An accompanying
numerical implementation and live in-browser simulator exercise
every formula below; the note stands on its own.

## Setup and notation {#setup-and-notation}

The intermediary retires $\lambda$ tonnes of carbon per unit time over
the horizon $[0, T]$. Each tonne requires $P$ kVCM tokens (the
protocol constant), purchased at the prevailing spot $S_t$. Retirement
demand is deterministic in this baseline; the token price is the sole
source of randomness. The kVCM spot is modelled as Geometric Brownian
Motion,

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t,
\qquad S_0 \text{ given (equal to } \pi_0 / P \text{ where } \pi_0
\text{ is the market carbon price per tonne at } t = 0\text{).}
$$

The carbon price per tonne $\pi_t := P \cdot S_t$ inherits the same
$(\mu, \sigma)$. The meet operator $a \wedge b := \min(a, b)$ recurs
below in stopping times.

*Price process.*

| Symbol | Meaning |
| --- | --- |
| $S_t$ | kVCM spot price at time $t$ (USD / kVCM) |
| $S_0$ | Initial kVCM spot at $t = 0$; $S_0 = \pi_0 / P$ |
| $\pi_0$ | Initial carbon price per tonne (USD) — market input |
| $\pi_t = P \cdot S_t$ | Carbon price per tonne (USD) |
| $\mu$ | GBM drift coefficient (annualised) |
| $\sigma$ | GBM volatility coefficient (annualised) |
| $W_t$ | Standard Brownian motion driving the GBM |
| $\mathcal{F}_t$ | Filtration — information available up to time $t$ |

*Contract and demand.*

| Symbol | Meaning |
| --- | --- |
| $P$ | Protocol constant — kVCM required per tonne |
| $\lambda$ | Retirement flow (tonnes / unit time), deterministic |
| $T$ | Horizon |
| $N = \lambda T$ | Total tonnes retired over $[0, T]$ |
| $f$ | Fee rate (e.g. $0.05$) |
| $Q$ | Fixed USD quote per tonne under the principal book |

*Treasury.*

| Symbol | Meaning |
| --- | --- |
| $k$ | Treasury's initial token notional (kVCM) |
| $C_{\mathrm{basis}}$ | Sunk USD cost of the treasury's opening position |
| $\alpha = \min(1, k / (N P))$ | Coverage fraction derived from $(k, P, \lambda, T)$ |
| $\tau_{\mathrm{cov}} = \min(T, k / (P \lambda))$ | Inventory-exhaustion time |

*Syndication.*

| Symbol | Meaning |
| --- | --- |
| $\beta \in [0, 1]$ | Ceded fraction of the b2b operating book |
| $\theta \ge 0$ | Counterparty risk load |
| $\pi_{\mathrm{syn}}$ | Up-front syndication premium (USD) |

*Switching.*

| Symbol | Meaning |
| --- | --- |
| $h \ge 1$ | Barrier multiple — switching book flips at $S_t = h \cdot S_0$ |
| $\tau$ | Switching stopping time |
| $f_{\mathrm{post}}$ | Fee rate applied after the switch fires |

*Jumps (compensated Merton overlay; details in §[Jumps](#compensated-merton-jump-diffusion)).*

| Symbol | Meaning |
| --- | --- |
| $\lambda_J$ | Poisson jump intensity (arrivals / yr) |
| $\mu_J$ | Mean log-jump size |
| $\sigma_J$ | Log-jump standard deviation |
| $\kappa = e^{\mu_J + \sigma_J^2 / 2} - 1$ | Merton compensation |

All P&L figures are **totals over $[0, T]$** in USD; divide by $N$ for
a per-tonne reading. $\mathbb{E}[\cdot]$, $\mathrm{Var}[\cdot]$,
$\mathrm{SD}[\cdot] := \sqrt{\mathrm{Var}[\cdot]}$,
$\mathrm{Cov}[\cdot, \cdot]$, and $\mathbb{P}[\cdot]$ carry their
standard probability-theory meanings.

### Risk metrics {#risk-metrics}

Tail-risk vocabulary used throughout the note and on every downstream
page:

$$
\mathrm{VaR}_p[\Pi] := -\inf\{ x : \mathbb{P}[\Pi \le x] \ge 1 - p \}
\qquad \text{(loss threshold at the $p$ quantile, reported as a positive USD figure),}
$$

$$
\mathrm{CVaR}_p[\Pi] := \mathbb{E}\!\left[ -\Pi \;\middle|\; \Pi \le -\mathrm{VaR}_p[\Pi] \right]
\qquad \text{(mean loss conditional on landing in the worst $1 - p$ fraction of paths),}
$$

$$
\mathrm{Sharpe}[\Pi] := \frac{\mathbb{E}[\Pi]}{\mathrm{SD}[\Pi]}
\qquad \text{(horizon-absolute, not annualised, not risk-free-rate adjusted),}
$$

$$
z := \frac{\mathbb{E}_{\mathrm{mc}}[\Pi] - \mathbb{E}_{\mathrm{cf}}[\Pi]}{\mathrm{stderr}}
\qquad \text{(MC-vs-closed-form sanity check; $|z| \le 2$ is sampling noise, $|z| > 3$ points at a bug).}
$$

The $\pm\mathrm{CI}_{95} = 1.96 \cdot \mathrm{SD} / \sqrt{n}$ half-width
accompanies every Monte-Carlo mean. Conventions specific to individual
books: $\mathrm{Sharpe}[R_{\mathrm{fee}}]$ is invariant in the fee rate
$f$, and $\mathrm{VaR}_p, \mathrm{CVaR}_p$ on the fee book read as
"low end of revenue" rather than loss ($R_{\mathrm{fee}} \ge 0$ by
construction).

### The kernel that runs through everything

Every operating book below reduces, up to a sign and scale, to an
integral of the GBM path. The treasury reduces to an integral over
the coverage window $[0, \tau_{\mathrm{cov}}]$ plus an optional
terminal-spot mark-to-market. The core kernel is

$$
I_T := \int_0^T S_t \, dt.
$$

Its first two moments are known in closed form (Dufresne 2001),

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

$I_T$ is not log-normal — only $S_t$ is — so tail quantiles (VaR,
CVaR) require Monte Carlo.

## The fee operating book {#fee-book}

The intermediary quotes clients $(1 + f) \cdot \pi_t$ and remits
$\pi_t$ to the protocol, keeping $f \cdot \pi_t$ per tonne. No
inventory, no capital. Total revenue over $[0, T]$ is

$$
R_{\mathrm{fee}}
  = \int_0^T f \cdot \pi_t \cdot \lambda \, dt
  = f \cdot P \cdot \lambda \cdot I_T,
$$

with

$$
\mathbb{E}[R_{\mathrm{fee}}] = f P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[R_{\mathrm{fee}}] = (f P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

Three consequences worth stating up front: $R_{\mathrm{fee}} \ge 0$
almost surely; top-line volatility is entirely driven by $\sigma$;
and the mean-to-SD ratio $\mathbb{E}[R_{\mathrm{fee}}] /
\mathrm{SD}[R_{\mathrm{fee}}]$ is invariant in $f$, so the fee rate is
a revenue lever but not a risk-adjusted-return lever.

## The back-to-back operating book {#back-to-back-book}

Fix a USD quote $Q$ at inception and source each tonne at spot as it
is retired. No capital, no opening position — the book starts at
$P\!\&\!L(0) = 0$:

$$
\Pi_{\mathrm{b2b}}
  = \int_0^T (Q - P \cdot S_t) \cdot \lambda \, dt
  = Q N - P \lambda \cdot I_T.
$$

Mean and variance inherit the kernel directly,

$$
\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q N - P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[\Pi_{\mathrm{b2b}}] = (P \lambda)^2 \cdot \mathrm{Var}[I_T],
$$

so $\mathrm{Var}[\Pi_{\mathrm{b2b}}] / \mathrm{Var}[R_{\mathrm{fee}}] =
(P/f)^2$ exactly — fee and b2b share the random kernel $I_T$, enter
it with opposite signs, and a single Monte Carlo pass prices both.
The payoff is linearly *decreasing* in $I_T$: upside is capped at
$Q N$ (the token collapsing to zero), downside is unbounded (the
token rallying). The book is economically equivalent to shorting a
continuous strip of forwards on kVCM struck at $Q / P$.

**Itô delta and the natural static hedge.** The Itô product rule on
the cumulative P&L gives the sensitivity of remaining expected P&L to
the current spot:

$$
\frac{\partial \, \mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi_{\mathrm{b2b}}(t) \mid \mathcal{F}_t]}{\partial S_t}
  = -P \lambda \cdot \frac{e^{\mu(T - t)} - 1}{\mu}
  \;\approx\; -P \lambda \cdot (T - t)
  \text{ for } \mu(T - t) \text{ small.}
$$

The fee book satisfies the same identity with a plus sign and
magnitude $f P \lambda (T - t)$: it is **long** kVCM beta, b2b is
**short**. The natural static hedge for a b2b position at time $t$
is therefore to hold $P \lambda (T - t)$ tokens of spot kVCM, which
is exactly what the treasury does when opened at $k = N P$ and
amortised through the horizon.

## The active treasury {#treasury-book}

The treasury is the intermediary's balance-sheet position. It opens
at $t = 0$ with $k$ tokens on hand at sunk basis $C_{\mathrm{basis}}$,
feeds retirement at spot over the coverage window, and marks any
over-hedge leftover at $S_T$:

$$
\tau_{\mathrm{cov}} := \min\!\left(T, \frac{k}{P \lambda}\right),
\qquad
k_{\mathrm{left}} := \max(0, k - N P).
$$

Terminal P&L is

$$
\Pi_{\mathrm{trea}}
  = P \lambda \int_0^{\tau_{\mathrm{cov}}} S_t \, dt
  + k_{\mathrm{left}} \cdot S_T
  - C_{\mathrm{basis}},
\qquad
\Pi_{\mathrm{trea}}(0) = k \cdot S_0 - C_{\mathrm{basis}}.
$$

Two branches, according to whether the horizon exhausts the
inventory:

**Under-/exactly hedged** ($k \le N P$, so $k_{\mathrm{left}} = 0$):

$$
\mathbb{E}[\Pi_{\mathrm{trea}}]
  = P \lambda \cdot S_0 \cdot \frac{e^{\mu \tau_{\mathrm{cov}}} - 1}{\mu}
  - C_{\mathrm{basis}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{trea}}]
  = (P \lambda)^2 \cdot \mathrm{Var}\!\left[\int_0^{\tau_{\mathrm{cov}}} S_t \, dt\right],
$$

with the inner variance given by the Dufresne identity above at
horizon $\tau_{\mathrm{cov}}$.

**Over-hedged** ($k > N P$, so $\tau_{\mathrm{cov}} = T$ and the
consumption integral equals $I_T$):

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

with $\mathrm{Var}[S_T] = S_0^2 \, e^{2\mu T} \, (e^{\sigma^2 T} - 1)$
and

$$
\mathrm{Cov}[I_T, S_T]
  = S_0^2 \cdot e^{\mu T} \cdot T \cdot
    \left( \frac{e^{(\mu + \sigma^2) T} - 1}{(\mu + \sigma^2) T}
         - \frac{e^{\mu T} - 1}{\mu T} \right),
$$

derived from $\mathbb{E}[S_t S_T] = S_0^2 \, e^{\mu(t + T) + \sigma^2
t}$ for $t \le T$ and integrated over $t \in [0, T]$. The treasury's
opening MTM $k \cdot S_0 - C_{\mathrm{basis}}$ translates the
distribution without reshaping it: the basis is a location
parameter, the token notional is a scale parameter on the
consumption integral.

## Strategy composition = operating + treasury {#strategy-composition}

The intermediary's **desk P&L** for any concrete strategy is

$$
\Pi_{\mathrm{desk}} = \Pi_{\mathrm{op}} + \Pi_{\mathrm{trea}},
$$

one operating book plus the treasury at the strategy's
$(k, C_{\mathrm{basis}})$. Canonical pairings recover every principal
variant in circulation:

| Strategy | Operating book | Treasury $(k, C_{\mathrm{basis}})$ |
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

The matched identity is the cleanest illustration of the split. At
$k = N P$, $C_{\mathrm{basis}} = N P S_0$,
$k_{\mathrm{left}} = 0$ and $\tau_{\mathrm{cov}} = T$, so

$$
\Pi_{\mathrm{matched}}
  = \Pi_{\mathrm{b2b}} + \Pi_{\mathrm{trea}}
  = (Q N - P \lambda I_T) + (P \lambda I_T - N P S_0)
  = N (Q - P S_0),
$$

**deterministic path-by-path**: $I_T$ cancels exactly between the
b2b operating book's stochastic leg and the treasury's consumption
of the full inventory. The `test/models.test.ts` suite asserts this
identity to machine precision.

### Partial-desk closed form

Let $J_\alpha := \int_{\alpha T}^T S_t \, dt$ be the uncovered-tail
integral. The partial-desk terminal P&L is

$$
\Pi_{\mathrm{partial}}
  = \Pi_{\mathrm{b2b}} + \Pi_{\mathrm{trea}}
  \bigl|_{(k, C_{\mathrm{basis}}) = (\alpha N P, \alpha N P S_0)}
  = Q N - \alpha N P S_0 - P \lambda \, J_\alpha.
$$

By the strong Markov property at $\alpha T$, $J_\alpha = S_{\alpha T}
\cdot Y$ where $Y := \int_0^{(1-\alpha) T} S'_s \, ds$ is an
independent unit-start GBM integral. Then

$$
\mathbb{E}[J_\alpha]
  = S_0 \, e^{\mu \alpha T}
    \cdot (1-\alpha) T \cdot \frac{e^{\mu (1-\alpha) T} - 1}{\mu (1-\alpha) T},
$$

$$
\mathrm{Var}[J_\alpha]
  = S_0^2 \, e^{(2 \mu + \sigma^2) \alpha T}
    \cdot \mathbb{E}[Y^2]
  - \bigl(S_0 \, e^{\mu \alpha T} \cdot \mathbb{E}[Y]\bigr)^2,
$$

with $\mathbb{E}[Y^2]$ given by the Dufresne identity at $S_0 = 1$
and horizon $(1-\alpha) T$. Therefore

$$
\mathbb{E}[\Pi_{\mathrm{partial}}]
  = Q N - \alpha N P S_0 - P \lambda \, \mathbb{E}[J_\alpha],
\qquad
\mathrm{Var}[\Pi_{\mathrm{partial}}]
  = (P \lambda)^2 \, \mathrm{Var}[J_\alpha].
$$

At $\alpha = 0$, $J_0 = I_T$ and the desk reduces to $\Pi_{\mathrm{b2b}}$.
At $\alpha = 1$, $J_1 = 0$ and the variance collapses to zero — the
matched identity. Between those endpoints, variance shrinks **not**
as $(1 - \alpha)^2 \mathrm{Var}[I_T]$ but as $\mathrm{Var}[J_\alpha]$,
which reflects horizon-based consumption rather than a convex
combination in tonnes.

## The syndicated-on-b2b operating book {#syndicated-book}

The treasury is an **internal** hedge: capital on the balance sheet
buys variance reduction on the full horizon. Syndication is the
complementary external primitive — cede a fraction $\beta$ of the
b2b operating book to a third-party counterparty against an up-front
premium $\pi_{\mathrm{syn}}$:

$$
\Pi_{\mathrm{ret}}
  = (1 - \beta) \cdot \Pi_{\mathrm{b2b}} + \pi_{\mathrm{syn}}.
$$

The book starts at $P\!\&\!L(0) = \pi_{\mathrm{syn}}$ (the
counterparty pays the premium at $t = 0$) and remains zero-capital
thereafter. Mean and variance inherit linearly:

$$
\mathbb{E}[\Pi_{\mathrm{ret}}]
  = (1 - \beta) \, \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{ret}}]
  = (1 - \beta)^2 \, \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Cession at the operating layer is α-free; the α scaling on variance
re-enters only when the retained book is composed with a treasury of
notional $\alpha N P$ at the desk level.

**Pricing the cession.** The per-unit risk load is

$$
\rho(\theta) =
\begin{cases}
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}]
  & \text{sharpe mode,} \\
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}]
    \cdot \phi(\Phi^{-1}(0.95)) / 0.05
  & \text{cvar mode,}
\end{cases}
$$

with $\phi, \Phi$ the standard normal density and CDF (shape factor
$\approx 2.063$). The loaded premium is

$$
\pi_{\mathrm{syn}}(\beta, \theta)
  = \beta \cdot
    \bigl(\mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta)\bigr).
$$

Both modes deliver a closed-form scalar, so the variance identity
above is **exact** under Monte Carlo. At $\theta = 0$,
$\pi_{\mathrm{syn}} = \beta \, \mathbb{E}[\Pi_{\mathrm{b2b}}]$ is the
actuarially fair premium; $\mathbb{E}[\Pi_{\mathrm{ret}}]$ is then
independent of $\beta$ — cession at the fair premium preserves
expected P&L. For $\theta > 0$ the counterparty charges a risk
premium, the intermediary sacrifices expected P&L, and tails tighten.

The `cvar` mode is a Gaussian-approximate surrogate — $I_T$'s true
tail is heavier than Gaussian — so Monte Carlo remains authoritative
for the retained tail. Tranched cessions — first-loss / senior splits
that price the ceded leg as $\max(L - K, 0)$ — are non-linear in
$I_T$, break the Dufresne backbone, and are out of scope here.

## The switching operating book {#switching-book}

The treasury and syndication *rescale* the loss tail. The switching
variant is the first primitive that **truncates** it. The intuition:
the b2b book's left tail is driven by paths on which $S_t$ rises
materially above $S_0$. If the intermediary flips pricing from the
fixed quote $Q$ to fee-book pricing at rate $f_{\mathrm{post}}$ the
first time $S_t$ crosses an upper barrier $H = h S_0$, every
post-barrier retirement earns the non-negative fee payoff and the
loss tail is capped.

Define the stopping time $\tau := \inf\{t \in [0, T] : S_t \ge H\}
\wedge T$ (with $\tau = T$ when the barrier is never reached) and
split the kernel,

$$
I_\tau := \int_0^\tau S_t \, dt,
\qquad
J_\tau := \int_\tau^T S_t \, dt,
\qquad
I_\tau + J_\tau = I_T.
$$

The switching operating book's P&L is

$$
\Pi_{\mathrm{sw}}
  = Q \lambda \tau
  - P \lambda I_\tau
  + f_{\mathrm{post}} \, P \lambda J_\tau
  \qquad \text{(b2b on $[0, \tau]$, fee book on $[\tau, T]$),}
$$

starting at $\Pi_{\mathrm{sw}}(0) = 0$. $h \to \infty$ gives $\tau =
T$ and recovers $\Pi_{\mathrm{b2b}}$; $h \le 1$ forces $\tau = 0$ and
the entire horizon runs as a fee book at rate $f_{\mathrm{post}}$.
The post-switch rate defaults to $f$ but is a free parameter.

**No closed-form density.** $\tau$ is a stopping time, not a path
integral, so $\Pi_{\mathrm{sw}}$ depends on the joint distribution of
$(\tau, I_\tau, J_\tau)$ — which has a non-trivial copula. Monte
Carlo is authoritative (see `src/core/simulate-switching.ts`).

**Pure-GBM closed-form anchors.** At $\lambda_J = 0$ the
Brownian-motion-with-drift hitting-time distribution (Harrison 1985;
Borodin-Salminen Table 3.0.1) gives

$$
\mathbb{P}[\tau \le T]
  = \Phi\!\left(\frac{-\log h + \nu T}{\sigma \sqrt{T}}\right)
  + h^{2\nu / \sigma^2} \cdot
    \Phi\!\left(\frac{-\log h - \nu T}{\sigma \sqrt{T}}\right),
\qquad \nu := \mu - \tfrac12 \sigma^2,
$$

with $\mathbb{E}[\tau \wedge T] = \int_0^T (1 - \mathbb{P}[\tau \le t])
\, dt$ as a tractable quadrature. The switching simulator is
regression-tested against these two scalars.

**Why a tail-truncation claim, not just a rescaling.** Partition
paths by whether the barrier fired:

$$
\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]
  \;\le\;
  \mathbb{P}[\tau = T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau = T\}}[\Pi_{\mathrm{b2b}}]
  + \mathbb{P}[\tau < T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau < T\}}[\text{post-switch fee}].
$$

The second term is non-negative almost surely — the post-switch fee
leg is revenue, bounded below by $0$. As $h \downarrow 1$ the first
term's weight shrinks and the bound tightens. The simulator reports
`CVaR95|no-switch` and `CVaR95|switched` separately so the truncation
is visible on the instrument, not just in prose.

**Operator decision surface.** Fix $(\mu, \sigma, f, f_{\mathrm{post}})$
and sweep $h$. $\mathbb{E}[\Pi_{\mathrm{sw}}]$ and
$\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]$ are monotone in $h$ in
opposite directions: a tighter barrier lifts the mean (fee revenue
replaces negative-skewed b2b exposure) and tightens the tail. The
knee against the b2b reference is the operator's decision point.
Optimal $h$ is a control-problem formulation (scope: Limitations).
The eyeball-the-knee presentation is deliberate: matching means does
not match distributions, and choosing a tail cap is a business
decision, not a pricing optimum.

## Direct comparison {#direct-comparison}

Moments of the four operating books plus the treasury, under pure
GBM:

| | Fee | B2b | Retained | Switching | Treasury $(k, C_{\mathrm{basis}})$ |
| --- | --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f P \lambda \mathbb{E}[I_T]$ | $Q N - P \lambda \mathbb{E}[I_T]$ | $(1 - \beta) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}}$ | MC only | $P \lambda \mathbb{E}[I_{\tau_{\mathrm{cov}}}] + k_{\mathrm{left}} S_0 e^{\mu T} - C_{\mathrm{basis}}$ |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \mathrm{Var}[I_T]$ | $(P \lambda)^2 \mathrm{Var}[I_T]$ | $(1 - \beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}]$ | MC only | branch on $k_{\mathrm{left}}$ (see §Treasury) |
| kVCM exposure | long | short | short, scaled by $1 - \beta$ | short on $[0, \tau]$, long on $[\tau, T]$ | long on $[0, \tau_{\mathrm{cov}}]$ plus $S_T$ exposure on leftover |
| Downside | bounded below by $0$ | unbounded | $(1 - \beta) \times$ b2b | bounded above on $\{\tau < T\}$ | $-C_{\mathrm{basis}}$ if $S \equiv 0$ |
| Capital | none | none | none | none | $C_{\mathrm{basis}}$ sunk |
| Counterparty | none | none | $\beta \cdot \Pi_{\mathrm{b2b}}$ upside | same as retained on the switched leg | none |

Desk totals (operating + treasury at the strategy's $k$,
$C_{\mathrm{basis}}$) are the row-wise sums.

### Break-even quote and the central asymmetry

Setting $\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$
solves for the quote at which the two operating books have identical
expected P&L:

$$
Q^* = (1 + f) \cdot P \cdot S_0 \cdot \frac{e^{\mu T} - 1}{\mu T}.
$$

As $\mu \to 0$, $Q^* \to (1 + f) \cdot P \cdot S_0$. Positive drift
pushes $Q^*$ above that level (b2b must charge extra to compensate
for expected appreciation); negative drift below.

**Matching means does not match distributions.** At $Q = Q^*$ fee
and b2b share the kernel $I_T$ and therefore share variance. But fee
enters $I_T$ with a plus sign (bounded below by $0$) while b2b
enters it with a minus sign (upside capped at $Q N$, downside
unbounded). The b2b book trades mean preservation for a left-skewed
loss tail the fee book does not have. This asymmetry — present even
at moment-equalised $Q$ — is the reason the treasury,
syndication, and switching dials exist.

## Jumps: the compensated Merton overlay {#compensated-merton-jump-diffusion}

The first relaxation of the baseline replaces pure GBM with Merton's
compensated jump-diffusion. The spot process becomes

$$
\frac{dS_t}{S_{t-}}
  = (\mu - \lambda_J \kappa) \, dt
  + \sigma \, dW_t
  + (J - 1) \, dN_t,
$$

where $N_t$ is a Poisson process of intensity $\lambda_J$ independent
of $W_t$, and at each arrival the price multiplies by $J = e^Y$ with
$Y \sim N(\mu_J, \sigma_J^2)$ i.i.d. The **Merton compensation**
$\kappa := \mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ subtracts
the jump component's mean effect, so the pure-$W$ part carries the
economic drift $\mu$.

**Means survive.** The exact solution is

$$
S_t = S_0 \exp\!\Big(
  (\mu - \tfrac12 \sigma^2 - \lambda_J \kappa) \, t
  + \sigma W_t
  + \sum_{k = 1}^{N_t} Y_k
\Big).
$$

Taking expectations and using the compound-Poisson identity
$\mathbb{E}\!\left[\prod_{k=1}^{N_t} e^{Y_k}\right]
  = e^{\lambda_J t (\mathbb{E}[e^Y] - 1)} = e^{\lambda_J \kappa t}$,

$$
\mathbb{E}[S_t]
  = S_0 \cdot e^{(\mu - \lambda_J \kappa) t}
    \cdot e^{\sigma^2 t / 2}
    \cdot e^{\lambda_J \kappa t}
  = S_0 \cdot e^{\mu t}
$$

— identical to the pure-GBM value for every $(\lambda_J, \mu_J, \sigma_J)$.
Integrating (Fubini),

$$
\mathbb{E}[I_T]
  = \int_0^T \mathbb{E}[S_t] \, dt
  = S_0 \cdot \frac{e^{\mu T} - 1}{\mu},
$$

so **every mean-level identity in this note carries over unchanged**:
$\mathbb{E}[R_{\mathrm{fee}}]$, $\mathbb{E}[\Pi_{\mathrm{b2b}}]$,
$\mathbb{E}[\Pi_{\mathrm{trea}}]$, $\mathbb{E}[\Pi_{\mathrm{ret}}]$,
and $Q^*$. The matched-desk identity $N(Q - P S_0)$ is pathwise
deterministic, so jumps leave it alone. The delta identities in the
b2b section pick up zero jump contribution in expectation and are
unchanged — with fatter realised hedging error.

**Variance does not.** Jumps inflate $\mathrm{Var}[S_t]$ and
therefore $\mathrm{Var}[I_T]$ and every downstream variance / tail
metric. A closed form exists but is more involved; the simulator
reports the pure-GBM $\mathrm{Var}[I_T]$ as a **GBM anchor** and
leaves jump-aware tail metrics to Monte Carlo. `test/jump-gbm.test.ts`
verifies the two predictions — means still match the GBM closed form
within CI, and the empirical variance strictly exceeds the GBM
anchor.

## Limitations and next steps {#limitations-and-next-steps}

| Baseline simplification | Removed in |
| --- | --- |
| Deterministic demand | Simulator — compound-Poisson order flow |
| GBM price dynamics | Simulator — compensated Merton jump-diffusion implemented (above); two-state regime switching pending |
| No historical calibration | Simulator — kVCM proxy (KLIMA, BCT, NCT) |
| No discounting, gas, or on-chain slippage | Simulator — parameterised |
| Passive treasury (no intraday rebalancing, over-hedge leftover marked at $S_T$) | Simulator — active consumption schedule is the baseline; dynamic delta hedge with perp/futures and tranched cessions pending |
| Static-rate syndication and switching | Simulator — quota-share syndication at the b2b operating layer; barrier-triggered switching one-way and non-adaptive; two-way and adaptive / optimal-$h$ pending |
| No credit / counterparty layer | Not scoped (syndication treats cession as default-free; tranching remains out of scope — non-linear in $I_T$, breaks the Dufresne backbone) |

This table is the authoritative scope statement; the Summary page
links here rather than restating it.

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Advances in Applied Probability, 33(1), 223–241. — closed-form moments of $I_T$.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.4. — simulation of path integrals of GBM.
- Harrison, J. M. (1985). *Brownian Motion and Stochastic Flow Systems.* — first-passage distribution used as the switching-book GBM test oracle.
