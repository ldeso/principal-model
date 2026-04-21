# principal-model

TypeScript simulator and Quarto report for the research note
*Klima Protocol — Fee-Based vs. Principal Model*
([`research-note.md`](research-note.md)).

Pages: **Summary** (`index.qmd`), **Model** (`model.qmd`), **Validation**
(`validation.qmd`), **Simulator** (`simulator.qmd`).

## Requirements

- Node.js ≥ 20
- npm ≥ 10
- Quarto ≥ 1.4 (report only)

## Install

```sh
npm install
```

## Run

```sh
npm run simulate -- --seed 42
npm run sweep    -- --seed 42
npm run typecheck
npm test
```

Artifacts land in `report/data/`:

- `run-<seed>.json` — single-run params, closed-form and MC metrics,
  $I_T$ histogram, sampled paths, P&L traces.
- `sweep.json` — $(\alpha, \mu, \sigma)$ grid, MC per cell.
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
| `--lambdaJ x` / `--muJ x` / `--sigmaJ x` | Merton jump params (0 ⇒ pure GBM) |
| `--h x` / `--fPost x` | threshold $h$, fee-mode rate ($h = \infty$ disables) |
| `--sweep` | also emit `sweep.json` |

## Report

```sh
npm run simulate -- --seed 42
npm run sweep    -- --seed 42
quarto preview report/validation.qmd
quarto preview report/simulator.qmd
```

## Layout

```
src/
  rng.ts                     Mulberry32 + Box-Muller + Knuth Poisson
  moments.ts                 Dufresne moments of I_T
  gbm.ts                     log-exact GBM + trapezoidal I_T + Merton overlay
  risk.ts                    quantile, VaR, CVaR, shortfall
  params.ts                  Params type and defaults
  models.ts                  closed form + MC for fee, b2b, retained, treasury
  report.ts                  scorecard + break-even + histograms
  cli.ts                     entrypoint
  fetch-historical-price.ts  Alchemy Prices pull
test/                        vitest suite
report/
  index.qmd                  Summary
  model.qmd                  Model (includes research-note.md)
  validation.qmd             Validation
  simulator.qmd              Simulator
  _glossary.qmd              shared glossary
  data/                      JSON artifacts
```

## Identities

- $\mathrm{Var}[\Pi_{\mathrm{b2b}}] / \mathrm{Var}[R_{\mathrm{fee}}] = (P / f)^2$.
- $\alpha = 1$: $\Pi_\alpha$ is deterministic.
- $\alpha = 0$: $\Pi_\alpha \equiv \Pi_{\mathrm{b2b}}$.
- $Q = Q^*$ equalises $\mathbb{E}[R_{\mathrm{fee}}]$ and $\mathbb{E}[\Pi_{\mathrm{b2b}}]$.
- $\mu = 0 \Rightarrow Q^* = (1 + f) \cdot P \cdot S_0$.
- Matched-book NAV drawdown: $\max_t [N P (1-t/T)(S_0 - S_t)]_+$
  (shortfall vs deterministic decay; see `src/risk.ts`).

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4.
