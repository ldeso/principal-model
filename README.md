# principal-model

`principal-model` simulates the books in
[`research-note.md`](research-note.md) — *Klima Protocol — Fee-Based
vs. Principal Model* — and builds the accompanying Quarto report. The
report has four pages: **Summary** (`index.qmd`), **Model**
(`model.qmd`), **Validation** (`validation.qmd`), and **Simulator**
(`simulator.qmd`).

## Getting started

The project needs Node.js ≥ 20 and npm ≥ 10; rendering the report also
needs Quarto ≥ 1.4. After `npm install`, the usual workflow is

```sh
npm install
npm run simulate -- --seed 42
npm run sweep    -- --seed 42
npm run typecheck
npm test
```

followed by `quarto preview report/validation.qmd` or `quarto preview
report/simulator.qmd` to see the rendered pages. The simulate and
sweep commands drop JSON artifacts into `report/data/`:
`run-<seed>.json` holds the single-run parameters together with the
closed-form and Monte Carlo metrics, the $I_T$ histogram, sampled
paths and P&L traces; `sweep.json` holds the $(\alpha, \mu, \sigma)$
grid with one Monte Carlo run per cell; and `qstar-surface.json` holds
the $Q^*(\mu, T)$ closed-form surface.

## Commands

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
