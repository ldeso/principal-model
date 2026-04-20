# Klima Protocol — Fee-Based vs. Principal Model

A research note on revenue and risk quantification for a
carbon-retirement intermediary operating against the Klima Protocol.

An intermediary buys carbon tokens (kVCM) on the open market and burns
them to retire carbon credits on behalf of clients. It can charge a
**fee** — a markup on the live token price — and pass the token cost
straight through; or it can quote a **fixed USD price** up front and
carry the inventory risk itself. This note quantifies what each choice
delivers in expected revenue and in tail risk, and when the two are
equivalent.

The note lays out the framework. Accompanying it are a numerical
implementation and a live in-browser simulator; the note stands on
its own.

## 1. Setup and notation

The intermediary retires $\lambda$ tonnes of carbon per unit time over
horizon $[0, T]$. Each tonne requires $P$ kVCM tokens (the protocol
constant), purchased at spot $S_t$. Retirement demand is deterministic
in this baseline; the sole source of randomness is the token price.

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
| $\alpha \in [0, 1]$ | Fraction of inventory pre-purchased (principal, generalised) |
| $\beta \in [0, 1]$ | Fraction of the $(1-\alpha)$ stochastic leg ceded (syndicated variant) |
| $\pi$ | Up-front premium received for ceding $\beta \cdot (1-\alpha)$ of the book |
| $\theta \geq 0$ | Counterparty risk-load multiplier (syndicated variant) |

**Price dynamics.** $S_t$ follows Geometric Brownian Motion under the
physical measure,

$$
dS_t = \mu \cdot S_t \cdot dt + \sigma \cdot S_t \cdot dW_t, \quad S_0 \text{ given,}
$$

so $\pi_t = P \cdot S_t$ is also GBM with the same $(\mu, \sigma)$.

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

$I_T$ is not log-normal, so tail metrics (VaR, CVaR) require Monte
Carlo; the closed-form moments serve as anchors.

## 2. Fee-based model (fee book)

The intermediary quotes clients $(1 + f) \cdot \pi_t$ and remits
$\pi_t$ to the protocol, keeping $f \cdot \pi_t$ per tonne. No
balance-sheet exposure, no inventory.

**Total revenue over $[0, T]$:**

$$
R_{\mathrm{fee}} = \int_0^T f \cdot \pi_t \cdot \lambda\,dt = f \cdot P \cdot \lambda \cdot I_T.
$$

**Moments:**

$$
\mathbb{E}[R_{\mathrm{fee}}] = f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T], \qquad
\mathrm{Var}[R_{\mathrm{fee}}] = (f \cdot P \cdot \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

**Properties.**

- $R_{\mathrm{fee}} \geq 0$ almost surely.
- Top-line volatility is fully driven by $\sigma$.
- Scales linearly in $f$, so the risk-adjusted return per unit fee is $f$-invariant.

## 3. Principal model

The intermediary fixes a USD quote $Q$ at $t = 0$. Five variants
follow, in order of increasing complexity: the first three set the
balance-sheet dial $\alpha$; the syndicated variant adds a counterparty dial $\beta$; the switching variant
adds a dynamic switch $h$.

### 3a. Matched (fully pre-bought, $\alpha = 1$)

Buy all $N \cdot P$ kVCM at inception for a fixed cost $C = N \cdot P
\cdot S_0$, then burn against demand over $[0, T]$. No terminal token
risk.

**Terminal P&L:**

$$
\Pi_{\mathrm{matched}} = N \cdot (Q - P \cdot S_0),
$$

which is **deterministic**.

Interim risk remains. The mark-to-market inventory value at time $t$
is

$$
V_t = (N \cdot P) \cdot (1 - t/T) \cdot S_t
$$

— a scaled GBM decayed by the burn schedule. Solvency, margin, or
covenant concerns live in the distribution of $\max_{t \leq T}
(V_0 - V_t)$.

### 3b. Back-to-back (buy at spot, $\alpha = 0$)

Quote $Q$ at inception but source each retirement at spot. The
intermediary is effectively short the token over $[0, T]$.

**Total P&L:**

$$
\Pi_{\mathrm{b2b}} = \int_0^T (Q - P \cdot S_t) \cdot \lambda\,dt = Q \cdot N - P \cdot \lambda \cdot I_T.
$$

**Moments:**

$$
\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T], \qquad
\mathrm{Var}[\Pi_{\mathrm{b2b}}] = (P \cdot \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

$\mathrm{Var}[\Pi_{\mathrm{b2b}}]$ equals $\mathrm{Var}[R_{\mathrm{fee}}]$
up to the rescaling $(P / f)^2$: the two books share **the same random
kernel** $I_T$, and a single Monte Carlo pass prices both.

**Payoff shape.** $\Pi_{\mathrm{b2b}}$ is linearly *decreasing* in
$I_T$. Upside caps at $Q \cdot N$ (reached as $S_t \to 0$); downside is
unbounded when kVCM rallies. Equivalent to shorting a continuous strip
of forwards on kVCM struck at $Q / P$.

### 3c. Partial ($\alpha$-pre-purchase, $\alpha \in [0, 1]$)

Pre-buy $\alpha \cdot N \cdot P$ kVCM at $S_0$; source the rest
back-to-back. $\alpha$ is the hedge ratio against spot.

$$
\begin{aligned}
\Pi_\alpha &= \alpha \cdot N \cdot (Q - P \cdot S_0) + (1 - \alpha) \cdot (Q \cdot N - P \cdot \lambda \cdot I_T) \\
&= Q \cdot N - P \cdot \lambda \cdot \left[ \alpha \cdot S_0 \cdot T + (1 - \alpha) \cdot I_T \right].
\end{aligned}
$$

The mean interpolates linearly in $\alpha$; the variance scales as
$(1-\alpha)^2$:

$$
\mathbb{E}[\Pi_\alpha] = (1 - \alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \cdot \Pi_{\mathrm{matched}}, \qquad
\mathrm{Var}[\Pi_\alpha] = (1 - \alpha)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

$\alpha = 1$ recovers the matched variant (deterministic); $\alpha = 0$ recovers the back-to-back variant.

### 3d. Syndicated (quota-share cession, $\beta \in [0, 1]$)

The partial variant is an **internal** hedge: capital on the balance sheet buys
variance reduction. The syndicated variant is the complementary primitive — sell part of
the stochastic leg to a third-party counterparty against an up-front
premium. No capital is tied up; the counterparty receives $\beta \cdot
(1-\alpha) \cdot \Pi_{\mathrm{b2b}}$ at $T$ against the premium paid at
$0$. Only the stochastic leg is ceded (syndicating the matched slice
is vacuous — zero variance).

**Retained P&L.**

$$
\Pi_{\mathrm{ret}} = \alpha \cdot N \cdot (Q - P \cdot S_0) + (1 - \alpha)(1 - \beta) \cdot \bigl( Q \cdot N - P \cdot \lambda \cdot I_T \bigr) + \pi.
$$

Still linear in $I_T$, so the Dufresne-moment backbone carries
through:

$$
\mathbb{E}[\Pi_{\mathrm{ret}}] = \alpha \cdot \Pi_{\mathrm{matched}} + (1 - \alpha)(1 - \beta) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \pi,
$$

$$
\mathrm{Var}[\Pi_{\mathrm{ret}}] = (1-\alpha)^2 (1-\beta)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Variance collapses as the *product* $(1-\alpha)^2(1-\beta)^2$: $\alpha$
and $\beta$ are multiplicatively equivalent on the variance scale but
orthogonal on the balance sheet — $\alpha$ consumes capital
$\alpha \cdot N \cdot P \cdot S_0$, $\beta$ consumes counterparty
credit.

**Premium.** The per-unit-cession risk load is

$$
\rho(\theta) = \begin{cases} \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] & \text{sharpe mode}, \\ \theta \cdot \mathrm{SD}[\Pi_{\mathrm{b2b}}] \cdot \phi\!\bigl(\Phi^{-1}(0.95)\bigr)/0.05 & \text{cvar mode}, \end{cases}
$$

with $\phi, \Phi$ the standard-normal density and CDF (the Gaussian
shape factor evaluates to $\approx 2.063$). The premium is

$$
\pi(\alpha, \beta, \theta) = \beta \cdot (1 - \alpha) \cdot \bigl( \mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta) \bigr).
$$

Both modes keep $\pi$ a closed-form scalar, so the retained-variance
identity is **exact** under Monte Carlo (no sample-moment dependence).
$\theta = 0$ gives the actuarially fair premium $\pi_{\mathrm{fair}} =
\beta(1-\alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}]$, at which
$\mathbb{E}[\Pi_{\mathrm{ret}}] = \mathbb{E}[\Pi_\alpha]$ independently
of $\beta$ — mean preservation across cession levels. $\theta > 0$ is
a risk premium the counterparty charges: the intermediary sacrifices
expected P&L for tighter tails.

The CVaR mode is a **Gaussian-approximate** surrogate — $I_T$ is not
log-normal and its true tail is heavier than Gaussian, so Monte Carlo
remains authoritative for the retained book's tail metrics. The shape
factor exists only to put `sharpe` and `cvar` modes on a common SD
scale.

**$Q^*$ is invariant in $\beta$.** $Q^*$ is defined by
$\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$, which
predates cession. The break-even quote is unaffected by how the
principal book is syndicated.

**Tranching is out of scope.** A first-loss / senior split would price
the ceded leg as $\max(L - K, 0)$ — non-linear in $I_T$, breaking the
Dufresne-moment backbone. It belongs in an MC-only note.

### 3e. Switching (barrier $h \geq 1$, post-switch rate $f_{\mathrm{post}}$)

The partial and syndicated variants **rescale** the loss tail by a constant factor — they
shrink it but cannot bound it. The switching variant is the first primitive that
**truncates** it. The intuition: the principal book's left tail is
driven by paths where $S_t$ rises materially above $S_0$. If the
intermediary flips pricing from the fixed quote $Q$ to the fee-book
markup $f_{\mathrm{post}} \cdot P \cdot S_t$ the first time $S_t$
crosses an upper barrier, every post-barrier loss is capped by
non-negative fee revenue.

**Stopping time and split integrals.** Fix $h \geq 1$, let $H = h
\cdot S_0$, and define

$$
\tau := \inf\{\,t \in [0, T] : S_t \ge H\,\} \;\wedge\; T,
$$

with $\tau = T$ when the barrier is never reached. Split the kernel:

$$
I_\tau := \int_0^\tau S_t\,dt, \qquad J_\tau := \int_\tau^T S_t\,dt, \qquad I_\tau + J_\tau = I_T.
$$

Under the switching rule the stochastic leg is

$$
\Pi_{\mathrm{sw}}^{(1-\alpha)} = Q \cdot \lambda \cdot \tau \;-\; P \cdot \lambda \cdot I_\tau \;+\; f_{\mathrm{post}} \cdot P \cdot \lambda \cdot J_\tau,
$$

— back-to-back up to $\tau$, fee-book revenue after. $h \to \infty$
gives $\tau = T$ and recovers $\Pi_{\mathrm{b2b}}$; $h \le 1$ forces
$\tau = 0$ and recovers the fee book at rate $f_{\mathrm{post}}$.
The post-switch rate defaults to $f$ but is a free parameter.

**Composition with $\alpha$ and $\beta$.** The switch touches only the
stochastic leg; the matched slice runs untouched and syndication
applies to the switched leg unchanged:

$$
\Pi_{3e} = \alpha \cdot N \cdot (Q - P \cdot S_0) + (1 - \alpha)(1 - \beta) \cdot \Pi_{\mathrm{sw}}^{(1-\alpha)} + \pi_{\mathrm{sw}}(\alpha, \beta, \theta).
$$

Three orthogonal levers: $\alpha$ consumes capital, $\beta$ consumes
counterparty credit, $h$ consumes a market-timing decision. They
commute under this scope convention.

**No closed-form density.** $\tau$ is a stopping time rather than a
path-average, so $\Pi_{3e}$ leaves the Dufresne family. It depends on
$(\tau, I_\tau, J_\tau)$ jointly, with a non-trivial copula: large
$I_\tau$ tends to precede barrier crossings; $J_\tau$ lives on a
random horizon $T - \tau$ starting at level $H$. Monte Carlo is
authoritative; the premium $\pi_{\mathrm{sw}}$ is computed from MC
moments (see `src/core/simulate-switching.ts`).

**Partial closed-form anchors under pure GBM.** Set $\lambda_J = 0$.
The Brownian-motion-with-drift hitting-time distribution (Harrison
1985; Borodin-Salminen Table 3.0.1) gives

$$
\mathbb{P}[\tau \le T] \;=\; \Phi\!\left(\tfrac{-\log h + \nu T}{\sigma \sqrt{T}}\right) + h^{2\nu/\sigma^2} \cdot \Phi\!\left(\tfrac{-\log h - \nu T}{\sigma \sqrt{T}}\right), \qquad \nu := \mu - \tfrac12 \sigma^2,
$$

with $\mathbb{E}[\tau \wedge T] = \int_0^T \bigl(1 - \mathbb{P}[\tau
\le t]\bigr)\,dt$ as a tractable quadrature. These two scalars are
**test oracles** for the switching simulator under pure GBM. Merton
jumps inflate the true $\tau$-distribution (a single jump can punch
through the barrier), so the GBM anchors become upper bounds for
$\mathbb{E}[\tau \wedge T]$ and lower bounds for $\mathbb{P}[\tau \le
T]$.

**Tail decomposition.** Partition paths by whether the barrier fired:

$$
\mathrm{CVaR}_{95}[\Pi_{3e}] \;\le\; \mathbb{P}[\tau = T] \cdot \mathrm{CVaR}_{95}^{\{\tau = T\}}[\Pi_{3b}] \;+\; \mathbb{P}[\tau < T] \cdot \mathrm{CVaR}_{95}^{\{\tau < T\}}[\text{post-switch fee}],
$$

where the second term is non-negative a.s. because the fee book is
bounded below by $0$. Equality is not exact (CVaR is not linear in
arbitrary partitions), but the inequality makes the claim "the barrier
truncates the loss tail" precise: the unswitched residual is bounded
by a vanilla back-to-back CVaR on its own measure; the switched residual is
bounded by a non-negative fee-book object. As $h \downarrow 1$ the first
term's weight shrinks and the bound tightens. The simulator reports
`CVaR95|no-switch` and `CVaR95|switched` separately for this reason.

**Operator decision surface.** Fix $(\mu, \sigma, f, f_{\mathrm{post}})$
and sweep $h$. $\mathbb{E}[\Pi_{3e}]$ and
$\mathrm{CVaR}_{95}[\Pi_{3e}]$ are monotone functions of $h$ in
opposite directions: a tighter barrier lifts the mean (fee revenue
replaces negative-skewed b2b exposure) and tightens the tail. The
curves' knee against the syndicated reference is the operator's decision
point. Optimal-$h$ is a control-problem formulation (scope: Limitations and next steps); the
eyeball-the-knee presentation is deliberate — matching means does not
match distributions, and choosing a tail cap is a business decision,
not a pricing optimum.

## 4. Risk quantification

For each book, the simulator reports:

| Metric | How |
| --- | --- |
| $\mathbb{E}[\Pi]$, $\mathrm{Var}[\Pi]$, $\mathrm{SD}[\Pi]$ | Closed form from the fee-based and principal models |
| VaR<sub>95</sub>, VaR<sub>99</sub> | Monte Carlo empirical quantile of $-\Pi$ |
| CVaR<sub>95</sub>, CVaR<sub>99</sub> | Monte Carlo tail mean of $-\Pi$ |
| $\mathbb{P}[\Pi < 0]$ | Monte Carlo |
| Sharpe-like $= \mathbb{E}[\Pi] / \mathrm{SD}[\Pi]$ | Closed form |
| Max NAV drawdown (matched variant only) | Monte Carlo on $V_t$ path |

### Itô dynamics and delta

Applying the Itô product rule to the cumulative P&L, the instantaneous
sensitivity of *remaining* P&L to $S_t$ is

$$
\text{Fee-based:} \quad \frac{\partial\,\mathbb{E}[R_{\mathrm{fee}} - R(t) \mid \mathcal{F}_t]}{\partial S_t} = f \cdot P \cdot \lambda \cdot \frac{e^{\mu(T-t)} - 1}{\mu} \approx f \cdot P \cdot \lambda \cdot (T - t) \text{ for } \mu T \text{ small,}
$$

$$
\text{Principal, back-to-back:} \quad \frac{\partial\,\mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi(t) \mid \mathcal{F}_t]}{\partial S_t} = -P \cdot \lambda \cdot \frac{e^{\mu(T-t)} - 1}{\mu} \approx -P \cdot \lambda \cdot (T - t).
$$

Signs are opposite: the fee book is **long** kVCM beta; the
back-to-back book is **short** kVCM beta. The matched book has zero
delta (fully pre-hedged by physical inventory).

The natural static hedge for the back-to-back book is to hold $(P
\cdot \lambda) \cdot (T - t)$ tokens of spot kVCM at time $t$ —
exactly the matched strategy amortised to the remaining horizon.

## 5. Direct comparison

| | Fee-based | Matched | Back-to-back | Syndicated | Switching |
| --- | --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T]$ | $N \cdot (Q - P \cdot S_0)$ | $Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T]$ | $\mathbb{E}[\Pi_\alpha] - \beta(1-\alpha)\rho(\theta)$ | MC only (no closed form) |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | $0$ (terminal) | $(P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | $(1-\alpha)^2(1-\beta)^2 (P \lambda)^2 \cdot \mathrm{Var}[I_T]$ | MC only; $\le \mathrm{Var}[\Pi_{\mathrm{ret}}]$ empirically |
| kVCM exposure | long | none (terminal), long (interim NAV) | short | short, scaled by $(1-\beta)$ | short on $[0, \tau]$, long on $[\tau, T]$ (fee leg) |
| Downside | bounded below by 0 | deterministic | unbounded | unbounded, scaled by $(1-\beta)$ | bounded above by $\lvert Q\lambda\tau - P\lambda I_\tau \rvert$ on $\{\tau < T\}$; back-to-back tail on $\{\tau = T\}$ |
| Capital requirement | none | $N \cdot P \cdot S_0$ | none | $\alpha \cdot N \cdot P \cdot S_0$ | $\alpha \cdot N \cdot P \cdot S_0$ (inherits from $\alpha$) |
| Counterparty exposure | none | none | none | $\beta \cdot (1-\alpha) \cdot P \cdot \lambda \cdot I_T$ upside (if counterparty defaults) | same as the syndicated variant on the switched leg |

### Break-even quote

The principal back-to-back book matches the fee book's *expected*
revenue when

$$
Q^* = (1 + f) \cdot P \cdot \mathbb{E}[I_T] / T = (1 + f) \cdot P \cdot S_0 \cdot \frac{e^{\mu T} - 1}{\mu T}.
$$

As $\mu \to 0$, $Q^* \to (1 + f) \cdot P \cdot S_0$ — the fee-based
time-zero quote. For $\mu > 0$ the principal model must quote *above*
that to compensate for expected kVCM appreciation; for $\mu < 0$,
below.

**Asymmetry observation.** *Matching means does not match
distributions.* At $Q = Q^*$ the two books share the kernel $I_T$ and
therefore share variance; but the fee book enters $I_T$ with a plus
sign (bounded below by $0$) while the back-to-back book enters it with
a minus sign (left-skewed loss tail). The principal book trades mean
preservation for a downside tail the fee book does not have.

## 6. Compensated Merton jump-diffusion

The first relaxation of the baseline replaces pure GBM with Merton's
jump-diffusion. The spot process becomes

$$
\frac{dS_t}{S_{t-}} = (\mu - \lambda_J \kappa)\,dt + \sigma\,dW_t + (J - 1)\,dN_t,
$$

where $N_t$ is a Poisson process with intensity $\lambda_J$
independent of $W_t$, and at each arrival the price multiplies by $J =
e^Y$ with $Y \sim N(\mu_J, \sigma_J^2)$ i.i.d. The drift offset
$\kappa := \mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ is the
**Merton compensation** — it subtracts the jump component's mean
effect so the pure-$W$ part carries the economic drift $\mu$.

### Means are invariant under compensation

The exact solution is

$$
S_t = S_0 \exp\!\Big(\big(\mu - \tfrac12 \sigma^2 - \lambda_J \kappa\big) t
      + \sigma W_t + \sum_{k = 1}^{N_t} Y_k\Big).
$$

Taking expectations and using independence of $W$, $N$, and
$\{Y_k\}$:

$$
\mathbb{E}[S_t]
  = S_0 \, e^{(\mu - \lambda_J \kappa) t}
    \, \mathbb{E}\!\left[e^{\sigma W_t}\right]
    \, \mathbb{E}\!\left[\prod_{k=1}^{N_t} e^{Y_k}\right]
  = S_0 \, e^{(\mu - \lambda_J \kappa) t} \cdot e^{\sigma^2 t/2}
    \cdot e^{\lambda_J \kappa t}
  = S_0 \, e^{\mu t},
$$

using the compound-Poisson generating-function identity
$\mathbb{E}\!\left[\prod_{k=1}^{N_t} e^{Y_k}\right]
  = e^{\lambda_J t (\mathbb{E}[e^Y] - 1)}
  = e^{\lambda_J \kappa t}$. $\mathbb{E}[S_t] = S_0 \cdot e^{\mu t}$
is therefore identical to the pure-GBM value for every
$(\lambda_J, \mu_J, \sigma_J)$.

Integrating over $[0, T]$ (Fubini):

$$
\mathbb{E}[I_T] = \int_0^T \mathbb{E}[S_t]\,dt = S_0 \cdot \frac{e^{\mu T} - 1}{\mu},
$$

so **Dufresne's GBM first-moment formula carries over unchanged**.
Every mean-level quantity linear in $I_T$ does too:

- $\mathbb{E}[R_{\mathrm{fee}}] = f \cdot P \cdot \lambda \cdot \mathbb{E}[I_T]$,
- $\mathbb{E}[\Pi_{\mathrm{b2b}}] = Q \cdot N - P \cdot \lambda \cdot \mathbb{E}[I_T]$,
- $\mathbb{E}[\Pi_\alpha] = (1-\alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \cdot \Pi_{\mathrm{matched}}$,
- $Q^* = (1+f) \cdot P \cdot S_0 \cdot (e^{\mu T} - 1)/(\mu T)$.

The matched variant's P&L is already pathwise-deterministic, so jumps leave it alone.

### Variance is not

Jumps inflate $\mathrm{Var}[S_t]$ (and therefore $\mathrm{Var}[I_T]$
and every downstream variance / tail metric). A closed form exists
but is more involved; the simulator reports the pure-GBM
$\mathrm{Var}[I_T]$ as a **GBM anchor** and leaves true jump-aware
tail metrics to Monte Carlo. The tests in `test/jump-gbm.test.ts`
confirm that the GBM closed-form means still match the jump-MC means
within CI, and that the empirical variance strictly exceeds the GBM
anchor.

### Itô deltas are unchanged in expectation

The jump contribution to $\mathbb{E}[S_t]$ is zero, so the
risk-quantification delta expressions and the matched variant's static hedge $(P \cdot \lambda) \cdot (T - t)$
carry over verbatim under compensated Merton — with fatter realised
hedging error.

## 7. Limitations and next steps

This table is the authoritative scope statement for the programme;
the landing page links here rather than duplicating it.

| Baseline simplification | Removed in |
| --- | --- |
| Deterministic demand | Simulator — compound-Poisson order flow |
| GBM price dynamics | Simulator — compensated Merton jump-diffusion implemented (see jump-diffusion section); two-state regime switching pending |
| No historical calibration | Simulator — kVCM proxy (KLIMA, BCT, NCT) |
| No discounting, gas, or on-chain slippage | Simulator — parameterised |
| Static (or absent) hedging | Simulator — dynamic delta hedge with inventory; perp/futures hedge if available; quota-share syndication is now in the syndicated variant; barrier-triggered mode switching is now in the switching variant (one-way and non-adaptive; two-way switching and adaptive/optimal-$h$ remain pending) |
| No credit / counterparty layer | Not scoped (the syndicated variant treats syndication as default-free; tranching remains out of scope — non-linear in $I_T$, would break the closed-form backbone) |

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Advances in Applied Probability, 33(1), 223–241. — closed-form moments of $I_T$.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.4. — simulation of path integrals of GBM.
- Harrison, J. M. (1985). *Brownian Motion and Stochastic Flow Systems.* — first-passage distribution used as the switching-variant GBM test oracle.
