// Empirical risk metrics on Monte Carlo P&L samples.
// Loss L := −Π; VaR_q and CVaR_q are reported as positive numbers.

export interface Summary {
  mean: number;
  variance: number;
  sd: number;
  stderr: number;
  ci95: number;
}

export function summarize(samples: ArrayLike<number>): Summary {
  const n = samples.length;
  if (n < 2) throw new Error("summarize: need at least 2 samples");
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i] as number;
  const mean = sum / n;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const d = (samples[i] as number) - mean;
    sse += d * d;
  }
  const variance = sse / (n - 1);
  const sd = Math.sqrt(variance);
  const stderr = sd / Math.sqrt(n);
  return { mean, variance, sd, stderr, ci95: 1.96 * stderr };
}

/** Linear-interpolation empirical quantile (numpy default). `q ∈ [0, 1]`. */
export function quantile(samples: ArrayLike<number>, q: number): number {
  if (q < 0 || q > 1) throw new Error("quantile: q must be in [0, 1]");
  const n = samples.length;
  if (n === 0) throw new Error("quantile: empty sample");
  const sorted = Float64Array.from(samples as ArrayLike<number>);
  sorted.sort();
  if (n === 1) return sorted[0] as number;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo] as number;
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

export function valueAtRisk(pnl: ArrayLike<number>, q: number): number {
  return -quantile(pnl, 1 - q);
}

export function conditionalVaR(pnl: ArrayLike<number>, q: number): number {
  const n = pnl.length;
  const sorted = Float64Array.from(pnl as ArrayLike<number>);
  sorted.sort();
  const k = Math.max(1, Math.ceil((1 - q) * n));
  let sum = 0;
  for (let i = 0; i < k; i++) sum += sorted[i] as number;
  return -(sum / k);
}

export function probLoss(pnl: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < pnl.length; i++) if ((pnl[i] as number) < 0) k++;
  return k / pnl.length;
}

/** Full per-sample P&L summary: mean/SD/95% CI, historical VaR/CVaR at 95
 *  and 99, loss probability, Sharpe. Bundles the metrics the OJS report
 *  cells need into one pass. Returns variance = 0 (and sharpe = null) for
 *  degenerate samples so callers don't have to special-case n < 2. */
export interface FullSummary {
  mean: number;
  sd: number;
  stderr: number;
  ci95: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  probLoss: number;
  sharpe: number | null;
}

export function summarise(samples: ArrayLike<number>): FullSummary {
  const n = samples.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i] as number;
  const mean = n > 0 ? sum / n : 0;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const d = (samples[i] as number) - mean;
    sse += d * d;
  }
  const variance = n > 1 ? sse / (n - 1) : 0;
  const sd = Math.sqrt(variance);
  const stderr = n > 0 ? sd / Math.sqrt(n) : 0;
  let losses = 0;
  for (let i = 0; i < n; i++) if ((samples[i] as number) < 0) losses++;
  return {
    mean,
    sd,
    stderr,
    ci95: 1.96 * stderr,
    var95: n > 0 ? valueAtRisk(samples, 0.95) : NaN,
    var99: n > 0 ? valueAtRisk(samples, 0.99) : NaN,
    cvar95: n > 0 ? conditionalVaR(samples, 0.95) : NaN,
    cvar99: n > 0 ? conditionalVaR(samples, 0.99) : NaN,
    probLoss: n > 0 ? losses / n : 0,
    sharpe: sd > 0 ? mean / sd : null,
  };
}

/** Matched-book NAV shortfall against the deterministic decay schedule:
 *   shortfall_t = max(0, N · P · (1 − k/N_steps) · (S_0 − S_k)).
 *
 * The research note's literal V_0 − V_t is pinned to V_0 by the V_T = 0 burn
 * convention and has zero variance; this schedule-relative version keeps the
 * same solvency intent with a non-trivial distribution. See README. */
export function shortfallVsSchedule(
  S: ArrayLike<number>,
  notional: number,
): number {
  const n = S.length;
  if (n === 0) return 0;
  const S0 = S[0] as number;
  let worst = 0;
  for (let k = 0; k < n; k++) {
    const decay = 1 - k / (n - 1);
    const gap = notional * decay * (S0 - (S[k] as number));
    if (gap > worst) worst = gap;
  }
  return worst;
}
