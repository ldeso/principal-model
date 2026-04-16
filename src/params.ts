// Typed parameter record for the simulator and a default instance.
//
// Symbols follow research-note.md §1.

export interface Params {
  /** kVCM/USD spot at t=0. */
  S0: number;
  /** GBM drift, annualised. */
  mu: number;
  /** GBM volatility, annualised. */
  sigma: number;
  /** Protocol price (kVCM per tonne), constant. */
  P: number;
  /** Retirement flow, tonnes / unit time. */
  lambda: number;
  /** Horizon in same time unit as μ, σ, λ. */
  T: number;
  /** Fee rate (e.g. 0.05 for 5%). */
  f: number;
  /** Fixed USD quote per tonne for the principal model. */
  Q: number;
  /** Pre-purchase fraction for 3c ∈ [0, 1]. α = 1 ↔ 3a; α = 0 ↔ 3b. */
  alpha: number;

  /** Number of Monte Carlo paths. */
  nPaths: number;
  /** Number of time steps per path. */
  nSteps: number;
  /** PRNG seed. */
  seed: number;
}

export const defaultParams: Params = {
  S0: 1.0,
  mu: 0.05,
  sigma: 0.5,
  P: 1.0,
  lambda: 1_000,
  T: 1.0,
  f: 0.05,
  Q: 1.08,
  alpha: 0.5,

  nPaths: 100_000,
  nSteps: 250,
  seed: 42,
};

/** Build a params record from `defaults` overlaid with `overrides`. */
export function withOverrides(
  defaults: Params,
  overrides: Partial<Params>,
): Params {
  return { ...defaults, ...overrides };
}
