# Klima Protocol: Fee-Based vs. Principal Model

A carbon-retirement intermediary buys kVCM tokens and burns them on
behalf of clients. The intermediary can charge clients in one of two
ways. A **fee** is a markup on the pass-through token cost; it carries
no inventory and no price exposure. A **principal** price is a fixed
USD quote per tonne, set at inception, and the intermediary absorbs
the spot risk.

This note has three parts. It derives closed-form moments for both
books. It splits the principal book into a zero-capital operating leg
and a balance-sheet treasury. It then studies three ways to reshape
the principal loss tail: pre-purchased inventory $(k, C_{\mathrm{basis}})$,
syndication at fraction $\beta$, and switching to fee mode above a
threshold $h$. The companion code checks every identity below.

## Setup and notation {#setup}

The intermediary retires $\lambda$ tonnes per unit time over $[0, T]$.
Each tonne needs $P$ kVCM, sourced at spot $S_t$. Demand is
deterministic, so the only random driver is the spot price. We model
$S_t$ as geometric Brownian motion,

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t,
\qquad S_0 = \pi_0 / P,
$$

and write $\pi_t := P S_t$ for the carbon price per tonne. Throughout,
$\Pi$ denotes a book's terminal P&L in USD over $[0, T]$. The
expectations $\mathbb{E}$, $\mathbb{P}$, $\mathrm{Var}$,
$\mathrm{SD}$, and $\mathrm{Cov}$ are taken under the GBM law above.
We write $\mathbf{1}\{A\}$ for the indicator of event $A$ and $a
\wedge b := \min(a, b)$.

The table below lists the price-process symbols.

| Symbol | Meaning |
| --- | --- |
| $S_t$ | kVCM spot (USD / kVCM) |
| $S_0$ | spot at $t = 0$; $S_0 = \pi_0 / P$ |
| $\pi_0$ | initial carbon price per tonne (USD) |
| $\pi_t = P S_t$ | carbon price per tonne |
| $\mu, \sigma$ | GBM drift and volatility (annualised) |
| $W_t$ | driving Brownian motion |
| $\mathcal{F}_t$ | natural filtration |

The next table lists the contract and demand symbols.

| Symbol | Meaning |
| --- | --- |
| $P$ | kVCM per tonne (protocol constant) |
| $\lambda$ | retirement flow, tonnes / unit time |
| $T$ | horizon |
| $N = \lambda T$ | total tonnes retired |
| $f$ | fee rate |
| $Q$ | fixed USD quote per tonne (principal) |

The treasury symbols describe a pre-purchased inventory of tokens.

| Symbol | Meaning |
| --- | --- |
| $k$ | treasury token notional (kVCM) |
| $C_{\mathrm{basis}}$ | sunk USD basis |
| $\alpha = \min(1, k / (N P))$ | coverage fraction |
| $\tau_{\mathrm{cov}} = \min(T, k / (P \lambda))$ | inventory-exhaustion time |

The syndication symbols describe a quota-share cession of the
principal book to a counterparty.

| Symbol | Meaning |
| --- | --- |
| $\beta \in [0, 1]$ | ceded fraction |
| $\theta \ge 0$ | counterparty risk load |
| $\pi_{\mathrm{syn}}$ | up-front premium (USD) |

The switching symbols describe the rule that flips the book between
fee mode and principal mode.

| Symbol | Meaning |
| --- | --- |
| $h \ge 1$ | threshold multiple |
| $H := h S_0$ | absolute threshold |
| $\mathbf{1}^{\mathrm{fee}}_t := \mathbf{1}\{S_t \ge H\}$ | fee-mode indicator |
| $\tau := \inf\{t : S_t \ge H\} \wedge T$ | first-passage time |
| $f_{\mathrm{post}}$ | fee rate in fee mode |

The last table lists the compensated Merton jump parameters,
introduced in §[Adding jumps](#jumps).

| Symbol | Meaning |
| --- | --- |
| $\lambda_J$ | Poisson intensity (/yr) |
| $\mu_J$ | mean log-jump |
| $\sigma_J$ | log-jump SD |
| $\kappa = e^{\mu_J + \sigma_J^2 / 2} - 1$ | Merton compensation |

P&L figures are totals over $[0, T]$ in USD. Divide by $N$ for a
per-tonne reading.

We report risk as value-at-risk and expected shortfall at 95 % and
99 %,

$$
\mathrm{VaR}_p[\Pi] := -\inf\{ x : \mathbb{P}[\Pi \le x] \ge 1 - p \},
\qquad
\mathrm{CVaR}_p[\Pi] := \mathbb{E}\!\left[ -\Pi \mid \Pi \le -\mathrm{VaR}_p[\Pi] \right],
$$

together with a horizon-absolute Sharpe ratio $\mathrm{Sharpe}[\Pi] :=
\mathbb{E}[\Pi] / \mathrm{SD}[\Pi]$. As a Monte Carlo cross-check, we
report the z-score of the MC mean against the closed-form mean,

$$
z := (\mathbb{E}_{\mathrm{mc}}[\Pi] - \mathbb{E}_{\mathrm{cf}}[\Pi]) / \mathrm{stderr}.
$$

We treat $|z| \le 2$ as sampling noise and $|z| > 3$ as suspect.
Monte Carlo means carry $\pm \mathrm{CI}_{95} = 1.96 \cdot \mathrm{SD}
/ \sqrt{n}$. Fee revenue is non-negative, so the fee book's VaR and
CVaR read as low-end revenue rather than loss, and its Sharpe does not
depend on $f$.

Every operating book below reduces, up to sign and scale, to the spot
integral

$$
I_T := \int_0^T S_t \, dt,
$$

for which Dufresne (2001) gives the moments

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

$I_T$ is not log-normal, so tail quantiles need Monte Carlo.

## The fee book {#fee-book}

The intermediary quotes each tonne at $(1 + f) \pi_t$, remits $\pi_t$
to the spot market, and keeps $f \pi_t$. Total revenue over the
horizon is therefore the fee rate times the spot integral,

$$
R_{\mathrm{fee}}
  = \int_0^T f \pi_t \lambda \, dt
  = f P \lambda \cdot I_T,
$$

and the first two moments follow from those of $I_T$,

$$
\mathbb{E}[R_{\mathrm{fee}}] = f P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[R_{\mathrm{fee}}] = (f P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

Revenue is non-negative almost surely. Variance is driven by $\sigma$
alone. The Sharpe ratio is invariant in $f$, since $f$ scales the
mean and the SD by the same factor.

## The back-to-back book {#b2b-book}

The intermediary fixes the quote $Q$ at inception and sources each
tonne at spot. Terminal P&L is the fixed payout minus the stochastic
sourcing cost,

$$
\Pi_{\mathrm{b2b}}
  = \int_0^T (Q - P S_t) \lambda \, dt
  = Q N - P \lambda \cdot I_T,
$$

so the moments are

$$
\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q N - P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[\Pi_{\mathrm{b2b}}] = (P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

The two books share the same random driver, so their variances
satisfy $\mathrm{Var}[\Pi_{\mathrm{b2b}}] / \mathrm{Var}[R_{\mathrm{fee}}] =
1 / f^2$ exactly, and one Monte Carlo pass prices both. Upside is
capped at $Q N$, and downside is unbounded. The position is
economically equivalent to shorting a continuous strip of forwards on
kVCM struck at $Q / P$.

The book's $S_t$-delta follows from Itô,

$$
\frac{\partial \, \mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi_{\mathrm{b2b}}(t) \mid \mathcal{F}_t]}{\partial S_t}
  = -P \lambda \cdot \frac{e^{\mu(T - t)} - 1}{\mu}
  \approx -P \lambda (T - t).
$$

The fee book satisfies the same identity with the sign reversed and
magnitude $f P \lambda (T - t)$. The natural static hedge for the
b2b book at time $t$ is therefore $P \lambda (T - t)$ tokens of spot
kVCM, which is exactly the treasury schedule at $k = N P$.

## The active treasury {#treasury-book}

The treasury opens at $t = 0$ with $k$ tokens at basis
$C_{\mathrm{basis}}$, feeds retirement at spot, and marks any
over-hedge leftover at the terminal spot $S_T$. Two derived quantities
organise the P&L. The inventory-exhaustion time
$\tau_{\mathrm{cov}} := \min(T, k / (P \lambda))$ is the date the
inventory runs out. The leftover stack
$k_{\mathrm{left}} := \max(0, k - N P)$ is the tokens still in the
treasury at $T$ when the initial inventory exceeds demand. Terminal
P&L is

$$
\Pi_{\mathrm{trea}}
  = P \lambda \int_0^{\tau_{\mathrm{cov}}} S_t \, dt
  + k_{\mathrm{left}} \cdot S_T
  - C_{\mathrm{basis}},
\qquad
\Pi_{\mathrm{trea}}(0) = k \cdot S_0 - C_{\mathrm{basis}}.
$$

When the treasury is under- or exactly hedged ($k \le N P$),

$$
\mathbb{E}[\Pi_{\mathrm{trea}}]
  = P \lambda \cdot S_0 \cdot \frac{e^{\mu \tau_{\mathrm{cov}}} - 1}{\mu}
  - C_{\mathrm{basis}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{trea}}]
  = (P \lambda)^2 \cdot \mathrm{Var}\!\left[\int_0^{\tau_{\mathrm{cov}}} S_t \, dt\right].
$$

When it is over-hedged ($k > N P$),

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

derived from $\mathbb{E}[S_t S_T] = S_0^2 e^{\mu(t + T) + \sigma^2 t}$
for $t \le T$, integrated over $[0, T]$. The opening MTM $k S_0 -
C_{\mathrm{basis}}$ translates the distribution without reshaping it.

## Strategies as operating plus treasury {#strategies}

Every desk in this note is an operating book plus a treasury,

$$
\Pi_{\mathrm{desk}} = \Pi_{\mathrm{op}} + \Pi_{\mathrm{trea}}.
$$

The table below names the desks and the treasury parameters that
recover each one. Each row pairs the operating book with the
$(k, C_{\mathrm{basis}})$ that, summed with it, gives the named desk.

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

At $(k, C_{\mathrm{basis}}) = (N P, N P S_0)$ the $I_T$ kernel cancels
path by path between the b2b operating leg and the treasury
consumption,

$$
\Pi_{\mathrm{matched}}
  = (Q N - P \lambda I_T) + (P \lambda I_T - N P S_0)
  = N (Q - P S_0).
$$

The matched desk is therefore deterministic, and the identity holds
to machine precision in `test/models.test.ts`.

Partial coverage sits between the naked and matched cases. Let
$J_\alpha := \int_{\alpha T}^T S_t \, dt$. Then

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

with $\mathbb{E}[Y^2]$ given by the Dufresne identity at $S_0 = 1$ and
horizon $(1-\alpha) T$. Hence

$$
\mathbb{E}[\Pi_{\mathrm{partial}}]
  = Q N - \alpha N P S_0 - P \lambda \, \mathbb{E}[J_\alpha],
\qquad
\mathrm{Var}[\Pi_{\mathrm{partial}}]
  = (P \lambda)^2 \, \mathrm{Var}[J_\alpha].
$$

At $\alpha = 0$, $J_0 = I_T$ and the desk is the naked b2b book; at
$\alpha = 1$, $J_1 = 0$ and the matched identity recovers. Between
them, variance decays with the length of the uncovered window, not
with $(1-\alpha)^2$.

## Syndicating the back-to-back book {#syndication}

The intermediary cedes a fraction $\beta$ of the b2b operating book
against an up-front premium $\pi_{\mathrm{syn}}$. The retained P&L is
therefore

$$
\Pi_{\mathrm{ret}}
  = (1 - \beta) \Pi_{\mathrm{b2b}} + \pi_{\mathrm{syn}},
\qquad
\Pi_{\mathrm{ret}}(0) = \pi_{\mathrm{syn}},
$$

and the moments scale linearly and quadratically in $(1 - \beta)$,

$$
\mathbb{E}[\Pi_{\mathrm{ret}}]
  = (1 - \beta) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}},
\qquad
\mathrm{Var}[\Pi_{\mathrm{ret}}]
  = (1 - \beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Cession acts on the operating layer alone and does not depend on
$\alpha$. The coverage fraction $\alpha$ re-enters later, when we
compose the operating book with the treasury into a desk.

The premium loads a per-unit risk charge $\rho(\theta)$ onto the
actuarially fair price. We support two loading modes: `sharpe` scales
the SD of the b2b book by $\theta$, and `cvar` replaces the SD by a
95 % tail-scaled SD through the Gaussian surrogate factor
$\phi(\Phi^{-1}(0.95)) / 0.05 \approx 2.063$,

$$
\rho(\theta) =
\begin{cases}
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}]
  & \text{sharpe,} \\
  \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] \cdot \phi(\Phi^{-1}(0.95)) / 0.05
  & \text{cvar.}
\end{cases}
$$

The loaded premium is the actuarial value of the ceded slice minus
this charge,

$$
\pi_{\mathrm{syn}}(\beta, \theta)
  = \beta \bigl(\mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta)\bigr).
$$

At $\theta = 0$ the premium is actuarially fair and
$\mathbb{E}[\Pi_{\mathrm{ret}}]$ does not depend on $\beta$. At
$\theta > 0$ the intermediary gives up expected P&L in exchange for
tail relief. The `cvar` mode is a Gaussian surrogate: the tail of
$I_T$ is heavier than Gaussian, so Monte Carlo remains authoritative.
Tranched cessions of the form $\max(L - K, 0)$ break the Dufresne
backbone and are out of scope.

## Switching to fee above a threshold {#switching}

The treasury and syndication rescale the loss tail. Switching cuts
it. Whenever the spot sits above the threshold $H = h S_0$, the book
quotes at the fee-mode rate $f_{\mathrm{post}}$ and accrues
non-negative revenue on that sub-interval. The mode indicator tracks
the spot symmetrically,

$$
M_t := \begin{cases}
  \text{fee} & S_t \ge H, \\
  \text{b2b} & S_t < H,
\end{cases}
\qquad
\mathbf{1}^{\mathrm{fee}}_t := \mathbf{1}\{S_t \ge H\}.
$$

We split the $I_T$ kernel and the occupation times by mode,

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

At the boundary, $h \to \infty$ recovers $\Pi_{\mathrm{b2b}}$, and
$h \le 1$ starts the book in fee mode.

With $\nu := \mu - \tfrac12 \sigma^2$ and $\log S_t \sim \mathcal{N}(\log S_0 + \nu t, \sigma^2 t)$,

$$
\mathbb{P}[S_t \ge H]
  = \Phi\!\left( \frac{\nu t - \log h}{\sigma \sqrt{t}} \right).
$$

The lognormal partial-expectation identity at $(m, v^2) = (\nu t,
\sigma^2 t)$ gives

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

Both integrals are Simpson-tractable and are implemented as
`expectedTimeAboveBarrier` and `expectedIntegralAboveBarrier` in
`src/core/moments.ts`. The first-passage time $\tau$ is a path
property. Under pure GBM it follows the Harrison / Borodin-Salminen
law,

$$
\mathbb{P}[\tau \le T]
  = \Phi\!\left(\frac{-\log h + \nu T}{\sigma \sqrt{T}}\right)
  + h^{2\nu / \sigma^2} \Phi\!\left(\frac{-\log h - \nu T}{\sigma \sqrt{T}}\right),
$$

with $\mathbb{E}[\tau \wedge T] = \int_0^T (1 - \mathbb{P}[\tau \le t]) \, dt$.
In general $T_{\mathrm{fee}} \ne T - \tau$, because the spot can
re-enter after its first crossing. The two quantities
$\mathbb{E}[\tau \wedge T]$ and $\mathbb{E}[T_{\mathrm{fee}}]$ are
therefore independent. The switching P&L density has no closed form,
so tail quantiles need Monte Carlo.

We can nonetheless bound the tail. Partitioning by whether the spot
ever crosses the threshold, we split the paths into the two events
$\{\tau = T\}$ (no crossing) and $\{\tau < T\}$ (at least one
crossing),

$$
\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]
  \le
  \mathbb{P}[\tau = T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau = T\}}[\Pi_{\mathrm{b2b}}]
  + \mathbb{P}[\tau < T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau < T\}}[\Pi_{\mathrm{sw}}].
$$

On $\{\tau = T\}$ the switching book coincides with the b2b book. On
$\{\tau < T\}$ every fee-mode sub-interval contributes non-negative
revenue. Lowering $h$ shrinks the first term and grows the second.
Re-entries temper the cut, because paths that cross back below $H$
resume b2b exposure. The simulator reports `CVaR95|no-switch` and
`CVaR95|switched` separately. Sweeping $h$ at fixed $(\mu, \sigma, f,
f_{\mathrm{post}})$ moves $\mathbb{E}[\Pi_{\mathrm{sw}}]$ and
$\mathrm{CVaR}_{95}[\Pi_{\mathrm{sw}}]$ monotonically in opposite
directions. Choosing $h$ to minimise a given risk measure is a
control problem and is not addressed here.

## Side-by-side comparison {#comparison}

The table below collects the pure-GBM results for each operating
book and the treasury, one book per column. Each column's $\Pi$ is
that book's terminal P&L, matching the earlier
$\Pi_{\mathrm{b2b}}$, $\Pi_{\mathrm{ret}}$, $\Pi_{\mathrm{sw}}$,
$\Pi_{\mathrm{trea}}$ ($R_{\mathrm{fee}}$ for the fee column). The
rows report, in order, the mean, the variance, the sign of the kVCM
exposure, the downside shape, the capital required, and the
counterparty exposure.

| | Fee | B2b | Retained | Switching | Treasury $(k, C_{\mathrm{basis}})$ |
| --- | --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f P \lambda \mathbb{E}[I_T]$ | $Q N - P \lambda \mathbb{E}[I_T]$ | $(1 - \beta) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi_{\mathrm{syn}}$ | MC; anchors on $\mathbb{E}[T_{\mathrm{fee}}]$, $\mathbb{E}[I_{\mathrm{fee}}]$ | $P \lambda \mathbb{E}[I_{\tau_{\mathrm{cov}}}] + k_{\mathrm{left}} S_0 e^{\mu T} - C_{\mathrm{basis}}$ |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \mathrm{Var}[I_T]$ | $(P \lambda)^2 \mathrm{Var}[I_T]$ | $(1 - \beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}]$ | MC | branch on $k_{\mathrm{left}}$ |
| kVCM exposure | long | short | short, $\times (1 - \beta)$ | short on $\{S_t < H\}$, long on $\{S_t \ge H\}$ | long on $[0, \tau_{\mathrm{cov}}]$, plus $S_T$ on leftover |
| Downside | $\ge 0$ | unbounded | $(1 - \beta) \times$ b2b | truncated on fee-mode intervals | $-C_{\mathrm{basis}}$ if $S \equiv 0$ |
| Capital | 0 | 0 | 0 | 0 | $C_{\mathrm{basis}}$ |
| Counterparty | none | none | $\beta \Pi_{\mathrm{b2b}}$ upside | as retained on the fee leg | none |

Reading one column gives the full profile of that book. A desk total
is the row-wise sum of an operating column and the treasury column,
as laid out in §[Strategies as operating plus treasury](#strategies).

Setting $\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$
solves for the break-even quote

$$
Q^* = (1 + f) P S_0 \cdot \frac{e^{\mu T} - 1}{\mu T},
$$

with $Q^* \to (1 + f) P S_0$ as $\mu \to 0$. A positive drift pushes
$Q^*$ above that level, and a negative drift pushes it below. At $Q
= Q^*$ the two books share $I_T$ and therefore share variance, but
they enter it with opposite signs. The fee book is bounded below.
The b2b book carries a left-skewed loss tail. This asymmetry
survives the moment equalisation at $Q = Q^*$ and is precisely what
the three dials of the principal book work on.

## Adding jumps {#jumps}

Replace pure GBM with

$$
\frac{dS_t}{S_{t-}}
  = (\mu - \lambda_J \kappa) \, dt
  + \sigma \, dW_t
  + (J - 1) \, dN_t,
$$

where $N_t$ is Poisson($\lambda_J$) independent of $W_t$, $J = e^Y$
and $Y \sim N(\mu_J, \sigma_J^2)$ i.i.d. The compensation $\kappa :=
\mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ removes the jump mean
from the drift.

Means survive the overlay. The compound-Poisson identity gives
$\mathbb{E}[S_t] = S_0 e^{\mu t}$ for every $(\lambda_J, \mu_J,
\sigma_J)$, so the mean of $I_T$ is still the GBM expression,

$$
\mathbb{E}[I_T] = S_0 \cdot \frac{e^{\mu T} - 1}{\mu}.
$$

Every mean-level identity above carries over unchanged, including
$\mathbb{E}[R_{\mathrm{fee}}]$, $\mathbb{E}[\Pi_{\mathrm{b2b}}]$,
$\mathbb{E}[\Pi_{\mathrm{trea}}]$, $\mathbb{E}[\Pi_{\mathrm{ret}}]$,
the break-even quote $Q^*$, and the matched-desk identity. Variances
do not. Jumps inflate $\mathrm{Var}[S_t]$, and with it
$\mathrm{Var}[I_T]$ and every downstream tail metric. The simulator
reports the pure-GBM variance as a GBM anchor and leaves the
jump-aware tails to Monte Carlo. The test `test/jump-gbm.test.ts`
verifies both predictions.

## Baselines this note assumes {#baselines}

The table below lists the simplifying assumptions this note makes
and points to where the simulator relaxes each one.

| Baseline | Lifted in |
| --- | --- |
| Deterministic demand | Simulator: compound-Poisson order flow |
| GBM dynamics | Simulator: compensated Merton (above); regime switching pending |
| No calibration | Simulator: kVCM proxy |
| No discounting / gas / slippage | Simulator: parameterised |
| Passive treasury | Simulator: active consumption schedule; dynamic delta hedge pending |
| Static syndication / switching | Simulator: quota-share cession; symmetric threshold; optimal-$h$ pending |
| No credit layer | Out of scope |

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Adv. Appl. Probab. 33(1), 223-241.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4.
- Harrison, J. M. (1985). *Brownian Motion and Stochastic Flow Systems.*
