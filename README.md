# principal-model

TypeScript simulator and interactive Quarto report for the research
note *Klima Protocol — Fee-Based vs. Principal Model*
([`research-note.md`](research-note.md)).

The report has four pages: **Summary** (index.qmd, findings +
drawable stress-test tool), **Model** (phase-a.qmd, the research
note), **Validation** (phase-b.qmd, closed-form vs. Monte Carlo
cross-check), and **Simulator** (phase-c.qmd, live in-browser Monte
Carlo). Every slider on the Simulator re-runs a fresh pass on the
parameters of your choosing (starting token price, starting carbon
price per tonne, drift and variance, optional Merton jump overlay,
initial inventory as tokens + cost basis, constant retirements per
day). A drawable custom-curve scenario on the Summary page (with an
Alchemy-fed historical preset) evaluates the three books on a single
user-sketched path.

## Requirements

- Node.js ≥ 20
- npm ≥ 10
- Quarto ≥ 1.4 (optional — only for the interactive report)

## Install

```sh
npm install
```

## Run

```sh
# One-shot run with defaults (see src/params.ts)
npm run simulate -- --seed 42

# Parameter sweep for the Observable sliders
npm run sweep -- --seed 42

# Type-check + unit tests
npm run typecheck
npm test
```

Artifacts are written to `report/data/`:

- `run-<seed>.json` — single-run: params, closed-form metrics, MC metrics,
  $I_T$ histogram, sampled paths, sub-sampled P&L traces.
- `sweep.json` — grid over $(\alpha, \mu, \sigma)$ with per-cell MC metrics.
- `qstar-surface.json` — $Q^*(\mu, T)$ closed-form surface.

### CLI flags

| flag | meaning |
| --- | --- |
| `--seed N` | PRNG seed |
| `--paths N` | override `nPaths` |
| `--steps N` | override `nSteps` |
| `--alpha x` | override $\alpha$ |
| `--mu x` / `--sigma x` | override $\mu$, $\sigma$ |
| `--f x` / `--Q x` / `--T x` | override fee, quote, horizon |
| `--lambdaJ x` / `--muJ x` / `--sigmaJ x` | Merton jump params (default 0 ⇒ pure GBM) |
| `--sweep` | additionally write `sweep.json` |

## Interactive report

```sh
# Validation page — pre-computed JSON, requires a simulate+sweep pass first.
npm run simulate -- --seed 42
npm run sweep    -- --seed 42
quarto preview report/phase-b.qmd

# Simulator page — live in-browser Monte Carlo, no JSON pre-pass needed.
quarto preview report/phase-c.qmd
```

## Layout

```
src/
  rng.ts                     seeded Mulberry32 + Box-Muller + Knuth Poisson
  moments.ts                 Dufresne E[I_T], Var[I_T] with μ→0 limit
  gbm.ts                     log-exact GBM stepper + trapezoidal I_T, optional Merton jumps
  risk.ts                    quantile, VaR, CVaR, shortfall-vs-schedule
  params.ts                  typed Params + defaults
  models.ts                  closed-form + MC for fee, 3a, 3b, 3c; break-even Q*
  report.ts                  scorecard + break-even table assembly + histograms
  cli.ts                     entrypoint — prints tables, writes JSON
  fetch-historical-price.ts  Alchemy Prices pull for the Summary page's historical preset
test/                        vitest unit + cross-check suite (43 tests)
report/
  index.qmd                  Summary — findings + drawable stress-test tool
  phase-a.qmd                Model — includes research-note.md
  phase-b.qmd                Validation — closed-form vs. Monte Carlo cross-check
  phase-c.qmd                Simulator — live in-browser Monte Carlo
  _glossary.qmd              shared glossary sidebar (included by every page)
  data/                      emitted JSON artifacts
```

## Verification notes

- $\mathrm{Var}[\Pi_{\mathrm{b2b}}] / \mathrm{Var}[R_{\mathrm{fee}}] = (P / f)^2$ — same $I_T$ kernel, rescaled (back-to-back book).
- $\alpha = 1$ collapses $\Pi_\alpha$ to the deterministic matched P&L path-by-path.
- $\alpha = 0$ makes $\Pi_\alpha$ coincide with $\Pi_{\mathrm{b2b}}$ path-by-path.
- $Q = Q^*$ equalises $\mathbb{E}[R_{\mathrm{fee}}]$ and $\mathbb{E}[\Pi_{\mathrm{b2b}}]$ (break-even quote).
- $\mu = 0 \Rightarrow Q^* = (1 + f) \cdot P \cdot S_0$ to machine precision.
- The matched-book NAV drawdown reports $\max_t [N \cdot P \cdot (1-t/T) \cdot (S_0 - S_t)]_+$ —
  shortfall against the deterministic decay schedule — because the
  literal $\max_t (V_0 - V_t)$ from the note is pinned to $V_0$ under
  the $V_T = 0$ burn convention. See `src/risk.ts` for details.

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.4.
