export interface Params {
  /** kVCM/USD spot at t=0. */
  S0: number;
  /** GBM drift (annualised). */
  mu: number;
  /** GBM volatility (annualised). */
  sigma: number;
  /** kVCM per tonne. */
  P: number;
  /** Retirement flow, tonnes / unit time. */
  lambda: number;
  /** Horizon. */
  T: number;
  /** Fee rate. */
  f: number;
  /** Fixed USD quote per tonne. */
  Q: number;
  /** Coverage fraction ∈ [0, 1]; α = 1 ↔ matched, α = 0 ↔ b2b. */
  alpha: number;

  /** Cession fraction of the b2b operating book. */
  beta: number;
  /** Counterparty risk load θ ≥ 0; 0 ⇒ fair premium. */
  premiumLoad: number;
  /** Load basis: `sharpe` ⇒ θ·SD; `cvar` ⇒ Gaussian-CVaR95 ≈ 2.063·SD. */
  premiumMode: "sharpe" | "cvar";

  /** Threshold ratio h = H/S₀; Infinity disables; h ≤ 1 starts in fee mode. */
  barrierRatio: number;
  /** Fee-mode fee rate; `null` ⇒ locked to `f`. */
  feePost: number | null;

  /** Merton intensity (/yr); 0 ⇒ pure GBM. */
  lambdaJ: number;
  /** Mean log-jump. */
  muJ: number;
  /** Log-jump SD. */
  sigmaJ: number;

  nPaths: number;
  nSteps: number;
  seed: number;
}

// μ, σ, λ_J, μ_J, σ_J: calibrated from report/data/kvcm-historical.json
// (5σ-bulk Merton split). S0, P, Q, f, λ, T, α at scale-free research-note
// values.
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
