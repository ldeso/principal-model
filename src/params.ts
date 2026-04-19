export interface Params {
  /** kVCM/USD spot at t=0. */
  S0: number;
  /** GBM drift (annualised). */
  mu: number;
  /** GBM volatility (annualised). */
  sigma: number;
  /** Protocol price kVCM/tonne, constant. */
  P: number;
  /** Retirement flow, tonnes per unit time. */
  lambda: number;
  /** Horizon (same time unit as μ, σ, λ). */
  T: number;
  /** Fee rate. */
  f: number;
  /** Fixed USD quote per tonne (principal model). */
  Q: number;
  /** Pre-purchase fraction for 3c ∈ [0, 1]; α = 1 ↔ 3a, α = 0 ↔ 3b. */
  alpha: number;

  /** Merton jump intensity (expected jumps per unit time). 0 ⇒ pure GBM. */
  lambdaJ: number;
  /** Mean of log-jump size. */
  muJ: number;
  /** SD of log-jump size. */
  sigmaJ: number;

  nPaths: number;
  nSteps: number;
  seed: number;
}

// Defaults for μ, σ, λ_J, μ_J, σ_J are calibrated from the kVCM daily series
// in report/data/kvcm-historical.json via a 5σ-bulk Merton split (the "low
// volume / low liquidity" regime of Phase A). S0, P, Q, f, λ, T, α remain at
// their scale-free research-note values so the closed-form identities in
// test/models.test.ts stay numerically convenient.
export const defaultParams: Params = {
  S0: 1.0,
  mu: -0.1,
  sigma: 0.25,
  P: 1.0,
  lambda: 1_000,
  T: 1.0,
  f: 0.05,
  Q: 1.08,
  alpha: 0.5,

  lambdaJ: 20,
  muJ: -0.05,
  sigmaJ: 0.35,

  nPaths: 100_000,
  nSteps: 250,
  seed: 42,
};

export function withOverrides(
  defaults: Params,
  overrides: Partial<Params>,
): Params {
  return { ...defaults, ...overrides };
}
