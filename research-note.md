# Klima Protocol — Fee-Based vs. Principal Model

A carbon-retirement intermediary buys kVCM tokens on the open market
and burns them to retire carbon credits on behalf of clients. Two
pricing choices sit at the root of the business. The **fee** book
passes the token cost through and adds a markup — no inventory, no
balance-sheet exposure. The **principal** book quotes a fixed USD
price up front and carries the inventory risk itself. This note
derives, in closed form, what each choice delivers in expected
revenue and in tail risk, and identifies three dials by which the
principal book's risk can be reshaped: pre-buying inventory
($\alpha$), syndicating part of the exposure ($\beta$), and
barrier-triggered switching back to the fee book ($h$). An
accompanying numerical implementation and live in-browser simulator
exercise every formula in this note; the note stands on its own.

## Setup and notation {#setup-and-notation}

The intermediary retires $\lambda$ tonnes of carbon per unit time over
the horizon $[0, T]$. Each tonne requires $P$ kVCM tokens (the
protocol constant), purchased at the prevailing spot $S_t$. Retirement
demand is deterministic in this baseline; the token price is the sole
source of randomness. The kVCM spot is modelled as Geometric Brownian
Motion,

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t, \qquad S_0 \text{ given.}
$$

The carbon price per tonne $\pi_t := P \cdot S_t$ inherits the same
$(\mu, \sigma)$.

| Symbol | Meaning |
| --- | --- |
| $S_t$ | kVCM spot price at time $t$ (USD / kVCM) |
| $P$ | Protocol constant — kVCM required per tonne |
| $\pi_t = P \cdot S_t$ | Carbon price per tonne (USD) |
| $\lambda$ | Retirement flow (tonnes / unit time), deterministic |
| $T$ | Horizon |
| $N = \lambda T$ | Total tonnes retired over $[0, T]$ |
| $f$ | Fee rate (e.g. $0.05$) |
| $Q$ | Fixed USD quote per tonne under the principal book |
| $\alpha \in [0, 1]$ | Pre-purchased fraction of inventory (coverage) |
| $\beta \in [0, 1]$ | Ceded fraction of the stochastic leg |
| $\theta \ge 0$ | Counterparty risk load |
| $\pi_{\mathrm{syn}}$ | Up-front syndication premium (USD) |
| $h \ge 1$ | Barrier multiple — the switching book flips at $S_t = h \cdot S_0$ |
| $\tau$ | Switching stopping time — first $t \in [0, T]$ with $S_t \ge h S_0$, else $T$ |
| $f_{\mathrm{post}}$ | Fee rate applied after the switch fires |

All P&L figures are **totals over $[0, T]$** in USD; divide by $N$ for
a per-tonne reading.

### The kernel that runs through everything

The six books introduced below all reduce, up to a sign and scale, to
integrals of the GBM path:

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
CVaR) require Monte Carlo. The Dufresne moments serve as anchors: any
stochastic-book mean or variance in this note is a scalar multiple of
$\mathbb{E}[I_T]$ or $\mathrm{Var}[I_T]$.

## The fee book {#fee-book}

The intermediary quotes clients $(1 + f) \cdot \pi_t$ and remits
$\pi_t$ to the protocol, keeping $f \cdot \pi_t$ per tonne. No
inventory, no balance-sheet exposure. Total revenue over $[0, T]$ is

$$
R_{\mathrm{fee}}
  = \int_0^T f \cdot \pi_t \cdot \lambda \, dt
  = f \cdot P \cdot \lambda \cdot I_T,
$$

with mean and variance

$$
\mathbb{E}[R_{\mathrm{fee}}] = f P \lambda \cdot \mathbb{E}[I_T],
\qquad
\mathrm{Var}[R_{\mathrm{fee}}] = (f P \lambda)^2 \cdot \mathrm{Var}[I_T].
$$

Three consequences worth stating up front: $R_{\mathrm{fee}} \ge 0$
almost surely (revenue cannot go negative); top-line volatility is
entirely driven by $\sigma$; and the mean-to-SD ratio
$\mathbb{E}[R_{\mathrm{fee}}] / \mathrm{SD}[R_{\mathrm{fee}}]$ is
invariant in $f$, so the fee rate is a revenue lever but not a
risk-adjusted-return lever.

## The principal book {#principal-book}

The intermediary fixes a USD quote $Q$ at $t = 0$ and commits to
retiring $N$ tonnes against it. Five variants follow, organised
around three orthogonal dials: $\alpha$ (what fraction of inventory
to pre-buy, consuming balance-sheet capital), $\beta$ (what fraction
of the remaining stochastic leg to cede, consuming counterparty
credit), and $h$ (a barrier at which pricing flips back to the fee
book, consuming a market-timing decision). Under the scope convention
that only the stochastic leg is affected by $\beta$ and $h$, the three
dials commute; each variant below is a special case of the full
three-dial composition.

### Matched ($\alpha = 1$) {#matched-book}

Buy all $N P$ kVCM at inception for $C = N P S_0$, then burn against
demand. Terminal P&L is

$$
\Pi_{\mathrm{matched}} = N \cdot (Q - P \cdot S_0),
$$

which is **deterministic**: whatever the path of $S_t$, the book pays
$C$ up front and receives $Q N$ over the horizon. Interim risk
survives, though. The mark-to-market inventory is
$V_t = N P \cdot (1 - t/T) \cdot S_t$ — a GBM amortised by the burn
schedule — and a crash between $0$ and $T$ shows up as a drawdown in
$V_t$ even though $V_T = 0$ pins the terminal accounts. Solvency,
margin, and covenant concerns live in the distribution of
$\max_{t \le T}(V_0 - V_t)$.

### Back-to-back ($\alpha = 0$) {#back-to-back-book}

Quote $Q$ at inception, source each tonne at spot. Effectively short
the token over $[0, T]$:

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
(P/f)^2$ exactly — fee and back-to-back share the random kernel
$I_T$, enter it with opposite signs, and a single Monte Carlo pass
prices both. The payoff is linearly *decreasing* in $I_T$: upside is
capped at $Q N$ (the token collapsing to zero), downside is unbounded
(the token rallying). The book is economically equivalent to shorting
a continuous strip of forwards on kVCM struck at $Q / P$.

**Itô delta and the natural static hedge.** The Itô product rule on
the cumulative P&L gives the sensitivity of remaining expected P&L to
the current spot:

$$
\frac{\partial \, \mathbb{E}[\Pi_{\mathrm{b2b}} - \Pi(t) \mid \mathcal{F}_t]}{\partial S_t}
  = -P \lambda \cdot \frac{e^{\mu(T - t)} - 1}{\mu}
  \;\approx\; -P \lambda \cdot (T - t)
  \text{ for } \mu(T - t) \text{ small.}
$$

The fee book satisfies the same identity with a plus sign and
magnitude $f P \lambda (T - t)$: it is **long** kVCM beta, the
back-to-back book is **short**, and the matched book has zero delta
because physical inventory has already absorbed it. The natural
static hedge for a back-to-back position at time $t$ is therefore to
hold $P \lambda (T - t)$ tokens of spot kVCM — which is precisely the
matched strategy amortised over the remaining horizon.

### Partial ($\alpha \in [0, 1]$) {#partial-book}

Pre-buy $\alpha N P$ kVCM at $S_0$; source the rest back-to-back. The
P&L interpolates linearly between the matched and back-to-back cases,

$$
\Pi_\alpha
  = \alpha N (Q - P S_0) + (1 - \alpha) \, (Q N - P \lambda I_T)
  = Q N - P \lambda \cdot \bigl[\alpha S_0 T + (1 - \alpha) I_T\bigr],
$$

with mean linear and variance quadratic in $1 - \alpha$,

$$
\mathbb{E}[\Pi_\alpha]
  = (1 - \alpha) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}]
  + \alpha \cdot \Pi_{\mathrm{matched}},
\qquad
\mathrm{Var}[\Pi_\alpha]
  = (1 - \alpha)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Pre-buying is a **linear dial** on SD and on downside quantiles like
CVaR: doubling coverage halves the tail loss. The endpoints
$\alpha = 0$ and $\alpha = 1$ recover back-to-back and matched
respectively. Raising coverage shifts mean by the fixed spread
$(\Pi_{\mathrm{matched}} - \mathbb{E}[\Pi_{\mathrm{b2b}}])$ per unit
$\alpha$, not by the sunk cost — the cost basis of pre-bought
inventory *translates* the distribution of matched-book P&L but does
not reshape its variance.

### Syndicated ($\beta \in [0, 1]$) {#syndicated-book}

$\alpha$ is an **internal** hedge: capital on the balance sheet buys
variance reduction. $\beta$ is the complementary external primitive —
cede a fraction $\beta$ of the stochastic leg to a third-party
counterparty against an up-front premium $\pi_{\mathrm{syn}}$. No
capital is tied up; the counterparty pays the premium at $t = 0$ and
receives $\beta (1 - \alpha) \Pi_{\mathrm{b2b}}$ at $T$. Only the
stochastic leg is ceded (syndicating the matched slice has nothing to
trade — zero variance already).

Retained P&L is

$$
\Pi_{\mathrm{ret}}
  = \alpha N (Q - P S_0)
  + (1 - \alpha)(1 - \beta) \cdot (Q N - P \lambda I_T)
  + \pi_{\mathrm{syn}},
$$

and is still linear in $I_T$, so the Dufresne-moment backbone carries
through:

$$
\mathbb{E}[\Pi_{\mathrm{ret}}]
  = \alpha \cdot \Pi_{\mathrm{matched}}
  + (1 - \alpha)(1 - \beta) \cdot \mathbb{E}[\Pi_{\mathrm{b2b}}]
  + \pi_{\mathrm{syn}},
$$

$$
\mathrm{Var}[\Pi_{\mathrm{ret}}]
  = (1 - \alpha)^2 (1 - \beta)^2 \cdot \mathrm{Var}[\Pi_{\mathrm{b2b}}].
$$

Variance collapses as the **product** $(1 - \alpha)^2 (1 - \beta)^2$:
$\alpha$ and $\beta$ are multiplicatively equivalent on the variance
scale but orthogonal on the balance sheet. $\alpha$ consumes
$\alpha N P S_0$ of capital; $\beta$ consumes counterparty credit.

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

with $\phi, \Phi$ the standard normal density and CDF (the shape
factor evaluates to $\approx 2.063$). The premium is

$$
\pi_{\mathrm{syn}}(\alpha, \beta, \theta)
  = \beta (1 - \alpha) \cdot
    \bigl(\mathbb{E}[\Pi_{\mathrm{b2b}}] - \rho(\theta)\bigr).
$$

Both modes deliver a closed-form scalar, so the variance identity
above is **exact** under Monte Carlo — no sample-moment dependence
sneaks in. At $\theta = 0$,
$\pi_{\mathrm{syn}} = \beta (1 - \alpha) \mathbb{E}[\Pi_{\mathrm{b2b}}]$
is the actuarially fair premium; at this value
$\mathbb{E}[\Pi_{\mathrm{ret}}]$ is independent of $\beta$ —
**cession at the fair premium preserves expected P&L**. For
$\theta > 0$ the counterparty charges a risk premium, the
intermediary sacrifices expected P&L, and tails tighten in return.

The `cvar` mode is a Gaussian-approximate surrogate — $I_T$'s true
tail is heavier than Gaussian — so Monte Carlo remains authoritative
for the retained tail; the shape factor exists only to put `sharpe`
and `cvar` modes on a common SD scale. $Q^*$ (introduced in §Direct
comparison) is defined *before* cession and is therefore invariant in
$\beta$. Tranched cessions — first-loss / senior splits that price
the ceded leg as $\max(L - K, 0)$ — are non-linear in $I_T$, break
the Dufresne backbone, and are out of scope here.

### Switching (barrier $h \ge 1$) {#switching-book}

$\alpha$ and $\beta$ *rescale* the loss tail — they shrink it but
cannot bound it. The switching variant is the first primitive that
**truncates** it. The intuition: the principal book's left tail is
driven by paths on which $S_t$ rises materially above $S_0$. If the
intermediary flips pricing from the fixed quote $Q$ to fee-book
pricing at rate $f_{\mathrm{post}}$ the first time $S_t$ crosses an
upper barrier $H = h S_0$, every post-barrier retirement earns the
non-negative fee payoff and the loss tail is capped.

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

The stochastic leg under the switching rule is

$$
\Pi_{\mathrm{sw}}^{(1 - \alpha)}
  = Q \lambda \tau
  - P \lambda I_\tau
  + f_{\mathrm{post}} P \lambda J_\tau
  \qquad \text{(back-to-back on $[0, \tau]$, fee book on $[\tau, T]$).}
$$

$h \to \infty$ gives $\tau = T$ and recovers $\Pi_{\mathrm{b2b}}$;
$h \le 1$ forces $\tau = 0$ and the entire horizon runs as a fee book
at rate $f_{\mathrm{post}}$. The post-switch rate defaults to $f$ but
is a free parameter. Composing with $\alpha$ and $\beta$,

$$
\Pi_{\mathrm{switching}}
  = \alpha N (Q - P S_0)
  + (1 - \alpha)(1 - \beta) \cdot \Pi_{\mathrm{sw}}^{(1 - \alpha)}
  + \pi_{\mathrm{syn}}.
$$

Three orthogonal levers: $\alpha$ consumes capital, $\beta$ consumes
counterparty credit, $h$ consumes a market-timing decision. They
commute under this scope convention.

**No closed-form density.** $\tau$ is a stopping time, not a path
integral, so $\Pi_{\mathrm{switching}}$ depends on the joint
distribution of $(\tau, I_\tau, J_\tau)$ — which has a non-trivial
copula: large $I_\tau$ values tend to precede barrier crossings, and
$J_\tau$ lives on a random residual horizon $T - \tau$ starting at
level $H$. Monte Carlo is authoritative; the premium
$\pi_{\mathrm{syn}}$ is computed from MC moments (see
`src/core/simulate-switching.ts`).

**Pure-GBM closed-form anchors.** At $\lambda_J = 0$ (no jumps), the
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
regression-tested against these two scalars. Under Merton jumps a
single jump can punch through the barrier, so the GBM anchors become
upper bounds for $\mathbb{E}[\tau \wedge T]$ and lower bounds for
$\mathbb{P}[\tau \le T]$.

**Why a tail-truncation claim, not just a rescaling.** Partition
paths by whether the barrier fired:

$$
\mathrm{CVaR}_{95}[\Pi_{\mathrm{switching}}]
  \;\le\;
  \mathbb{P}[\tau = T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau = T\}}[\Pi_{\mathrm{b2b}}]
  + \mathbb{P}[\tau < T] \cdot
    \mathrm{CVaR}_{95}^{\{\tau < T\}}[\text{post-switch fee}].
$$

The second term is non-negative almost surely — the post-switch fee
leg is revenue, bounded below by $0$. As $h \downarrow 1$ the first
term's weight shrinks and the bound tightens. The simulator reports
`CVaR95|no-switch` and `CVaR95|switched` separately so the
truncation is visible on the instrument, not just in prose.

**Operator decision surface.** Fix $(\mu, \sigma, f, f_{\mathrm{post}})$
and sweep $h$. $\mathbb{E}[\Pi_{\mathrm{switching}}]$ and
$\mathrm{CVaR}_{95}[\Pi_{\mathrm{switching}}]$ are monotone in $h$ in
opposite directions: a tighter barrier lifts the mean (fee revenue
replaces negative-skewed b2b exposure) and tightens the tail. The
knee of these two curves against the syndicated reference is the
operator's decision point. Optimal $h$ is a control-problem formulation
(scope: Limitations). The eyeball-the-knee presentation is
deliberate: matching means does not match distributions, and choosing
a tail cap is a business decision, not a pricing optimum.

## Direct comparison {#direct-comparison}

| | Fee | Matched | Back-to-back | Partial | Syndicated | Switching |
| --- | --- | --- | --- | --- | --- | --- |
| $\mathbb{E}[\Pi]$ | $f P \lambda \mathbb{E}[I_T]$ | $N (Q - P S_0)$ | $Q N - P \lambda \mathbb{E}[I_T]$ | $(1{-}\alpha) \mathbb{E}[\Pi_{\mathrm{b2b}}] + \alpha \Pi_{\mathrm{matched}}$ | $\mathbb{E}[\Pi_\alpha] - \beta (1{-}\alpha) \rho(\theta)$ | MC only |
| $\mathrm{Var}[\Pi]$ | $(f P \lambda)^2 \mathrm{Var}[I_T]$ | $0$ (terminal) | $(P \lambda)^2 \mathrm{Var}[I_T]$ | $(1{-}\alpha)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}]$ | $(1{-}\alpha)^2 (1{-}\beta)^2 \mathrm{Var}[\Pi_{\mathrm{b2b}}]$ | MC only |
| kVCM exposure | long | none terminal, long interim | short | partial short | short, scaled by $1{-}\beta$ | short on $[0, \tau]$, long on $[\tau, T]$ |
| Downside | bounded below by $0$ | deterministic | unbounded | $(1{-}\alpha) \times$ back-to-back | $(1{-}\alpha)(1{-}\beta) \times$ back-to-back | bounded above on $\{\tau < T\}$ |
| Capital | none | $N P S_0$ | none | $\alpha N P S_0$ | $\alpha N P S_0$ | inherits from $\alpha$ |
| Counterparty | none | none | none | none | $\beta (1{-}\alpha) P \lambda I_T$ upside | same as syndicated on the switched leg |

### Break-even quote and the central asymmetry

Setting $\mathbb{E}[R_{\mathrm{fee}}] = \mathbb{E}[\Pi_{\mathrm{b2b}}]$
solves for the quote at which the two books have identical expected
P&L:

$$
Q^* = (1 + f) \cdot P \cdot S_0 \cdot \frac{e^{\mu T} - 1}{\mu T}.
$$

As $\mu \to 0$, $Q^* \to (1 + f) \cdot P \cdot S_0$ — the fee-based
time-zero quote. Positive drift pushes $Q^*$ above that level (the
principal book must charge extra to compensate for expected
appreciation); negative drift below.

**Matching means does not match distributions.** At $Q = Q^*$ the fee
and back-to-back books share the kernel $I_T$ and therefore share
variance. But the fee book enters $I_T$ with a plus sign (bounded
below by $0$) while the back-to-back book enters it with a minus
sign (upside capped at $Q N$, downside unbounded). The principal
book trades mean preservation for a left-skewed loss tail the fee
book does not have. This asymmetry — present even at
moment-equalised $Q$ — is the reason the $\alpha, \beta, h$ dials
exist.

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
$\mathbb{E}[\Pi_\alpha]$, and $Q^*$. The matched book's P&L is
pathwise deterministic, so jumps leave it alone. The delta identities
in the back-to-back section pick up zero jump contribution in
expectation and are also unchanged — with fatter realised hedging
error.

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
| Static (or absent) hedging | Simulator — dynamic delta hedge with inventory; perp/futures hedge if available; quota-share syndication live in the syndicated book; barrier-triggered mode switching live in the switching book (one-way, non-adaptive; two-way and adaptive / optimal-$h$ pending) |
| No credit / counterparty layer | Not scoped (the syndicated book treats cession as default-free; tranching remains out of scope — non-linear in $I_T$, breaks the Dufresne backbone) |

This table is the authoritative scope statement; the Summary page
links here rather than restating it.

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.* Advances in Applied Probability, 33(1), 223–241. — closed-form moments of $I_T$.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.4. — simulation of path integrals of GBM.
- Harrison, J. M. (1985). *Brownian Motion and Stochastic Flow Systems.* — first-passage distribution used as the switching-book GBM test oracle.
