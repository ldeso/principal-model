// Empirical risk metrics on Monte Carlo P&L samples.
//
// Conventions follow the §4 table in research-note.md:
//   Loss L := −Π. VaR_q is the q-quantile of L; CVaR_q is E[L | L ≥ VaR_q].
//   Both are reported as positive numbers (larger = worse).

export interface Summary {
  mean: number;
  variance: number;
  sd: number;
  /** Standard error of the mean estimate: sd / sqrt(n). */
  stderr: number;
  /** 95% CI half-width on the mean: 1.96 · stderr. */
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

/** Linear-interpolation empirical quantile (numpy default). `q ∈ [0, 1]`.
 * Mutates a local copy; the input is untouched. */
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

/** VaR_q of a P&L sample: positive when there are losses in the q-tail. */
export function valueAtRisk(pnl: ArrayLike<number>, q: number): number {
  // Loss is −Π; VaR_q is the q-quantile of loss, i.e. the (1−q)-quantile of Π,
  // negated so positive VaR ↔ loss.
  return -quantile(pnl, 1 - q);
}

/** CVaR_q: average loss conditional on loss ≥ VaR_q. */
export function conditionalVaR(pnl: ArrayLike<number>, q: number): number {
  const n = pnl.length;
  const sorted = Float64Array.from(pnl as ArrayLike<number>);
  sorted.sort();
  // Worst (1−q) fraction of outcomes sit at the low end of sorted Π.
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

/** Max shortfall of the realised NAV against the deterministic decay
 * schedule for the §3a matched book:
 *   shortfall_t = max(0, V_0 · (1 − t/T) − V_t)
 *              = max(0, N · P · (1 − k/N_steps) · (S_0 − S_k)).
 *
 * The research note's literal "V_0 − V_t" definition is degenerate here
 * because V_T = 0 by the burn schedule, pinning the max to V_0 with zero
 * variance. The shortfall-against-schedule version captures the same
 * solvency/mark-to-market intent while keeping a non-trivial distribution
 * — zero when S_t tracks or exceeds S_0 on the scaled curve, positive
 * otherwise. Returned as a non-negative dollar amount. */
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
