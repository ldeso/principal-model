# principal-model — Phase B

TypeScript simulator and interactive report for the research note *Klima
Protocol — Fee-Based vs. Principal Model* (`research-note.md`, Phase A).

This repository covers **Phase B**: reproduce the §1–§5 closed-form
quantities computationally, estimate the §4 tail-risk metrics by Monte
Carlo, and expose the results through a Quarto + Observable notebook.
Phase C extensions (gas, slippage, jumps, calibration, dynamic hedging)
are out of scope.

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
  I_T histogram, sampled paths, sub-sampled P&L traces.
- `sweep.json` — grid over (α, μ, σ) with per-cell MC metrics.
- `qstar-surface.json` — Q\*(μ, T) closed-form surface.

### CLI flags

| flag | meaning |
| --- | --- |
| `--seed N` | PRNG seed |
| `--paths N` | override `nPaths` |
| `--steps N` | override `nSteps` |
| `--alpha x` | override α |
| `--mu x` / `--sigma x` | override μ, σ |
| `--f x` / `--Q x` / `--T x` | override fee, quote, horizon |
| `--sweep` | additionally write `sweep.json` |

## Interactive report

```sh
quarto preview report/report.qmd
```

Regenerate the JSON (`npm run simulate` + `npm run sweep`) before previewing.

## Layout

```
src/
  rng.ts       seeded Mulberry32 + Box-Muller
  moments.ts   Dufresne E[I_T], Var[I_T] with μ→0 limit
  gbm.ts       log-exact GBM stepper + trapezoidal I_T
  risk.ts     quantile, VaR, CVaR, shortfall-vs-schedule
  params.ts    typed Params + defaults
  models.ts    closed-form + MC for fee, 3a, 3b, 3c; break-even Q*
  report.ts    §4/§5 table assembly + histograms
  cli.ts       entrypoint — prints tables, writes JSON
test/          vitest unit + cross-check suite (39 tests)
report/
  report.qmd   Quarto + Observable notebook
  data/        emitted JSON artifacts
```

## Verification notes

- `Var[Π_b2b] / Var[R_fee] = (P / f)²` — same I_T kernel, rescaled (§3b).
- `α = 1` collapses Π_α to the deterministic matched P&L path-by-path.
- `α = 0` makes Π_α coincide with Π_b2b path-by-path.
- `Q = Q*` equalises `E[R_fee]` and `E[Π_b2b]` (§5).
- `μ = 0` ⇒ `Q* = (1 + f) · P · S_0` to machine precision.
- The §3a NAV drawdown reports `max_t [N·P·(1−t/T)·(S_0 − S_t)]+` —
  shortfall against the deterministic decay schedule — because the
  literal `max_t (V_0 − V_t)` from the note is pinned to V_0 under
  the `V_T = 0` burn convention. See `src/risk.ts` for details.

## References

- Dufresne, D. (2001). *The integral of geometric Brownian motion.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, §3.4.
