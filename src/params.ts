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
  /** Pre-purchase fraction for the partial variant ∈ [0, 1]; α = 1 ↔ matched, α = 0 ↔ back-to-back. */
  alpha: number;

  /** Syndicated-variant external cession fraction of the (1−α) stochastic leg; 0 ⇒ no
   *  syndication and Π_ret ≡ Π_α. */
  beta: number;
  /** Syndicated-variant counterparty risk-load multiplier θ ≥ 0. θ = 0 ⇒ fair premium. */
  premiumLoad: number;
  /** Syndicated-variant load basis: `"sharpe"` multiplies θ·SD[Π_b2b]; `"cvar"` multiplies
   *  the Gaussian-CVaR95 proxy ≈ 2.063·SD[Π_b2b]. */
  premiumMode: "sharpe" | "cvar";

  /** Switching-variant upper barrier ratio h = H/S0. Switch from b2b to fee the first time
   *  S_t ≥ h·S0. h = Infinity disables the switch (falls back to the syndicated retained book);
   *  h ≤ 1 fires the switch at t = 0. */
  barrierRatio: number;
  /** Switching-variant post-switch fee rate applied to the tail integral J_τ. `null` locks
   *  it to `f` (zero-config path); a number sets it independently for the
   *  "richer markup after the flip" research lever. */
  feePost: number | null;

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
// volume / low liquidity" regime of the model). S0, P, Q, f, λ, T, α remain
// at their scale-free research-note values so the closed-form identities in
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

  beta: 0,
  premiumLoad: 0,
  premiumMode: "sharpe",

  barrierRatio: Infinity,
  feePost: null,

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
