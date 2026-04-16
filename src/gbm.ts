// Exact GBM path simulator and I_T quadrature.
//
// Uses the log-exact scheme
//   S_{t+Δ} = S_t · exp((μ − σ²/2) Δ + σ √Δ · Z), Z ~ N(0,1),
// so the terminal distribution is unbiased for any step count. The
// path integral I_T = ∫₀ᵀ S_t dt is estimated by the trapezoidal rule
// over the N_steps+1 sampled nodes. Quadrature error is O(Δ²).

import type { Rng } from "./rng.ts";

export interface GbmPath {
  /** Sampled prices at t_0 = 0, t_1 = Δ, …, t_N = T. Length N_steps + 1. */
  S: Float64Array;
  /** Trapezoidal estimate of ∫₀ᵀ S_t dt. */
  IT: number;
}

export interface SamplePathOpts {
  S0: number;
  mu: number;
  sigma: number;
  T: number;
  nSteps: number;
}

export function samplePath(rng: Rng, opts: SamplePathOpts): GbmPath {
  const { S0, mu, sigma, T, nSteps } = opts;
  const dt = T / nSteps;
  const drift = (mu - 0.5 * sigma * sigma) * dt;
  const diffusion = sigma * Math.sqrt(dt);

  const S = new Float64Array(nSteps + 1);
  S[0] = S0;

  // Trapezoidal sum: Δ · (½ S_0 + S_1 + … + S_{N−1} + ½ S_N).
  let acc = 0.5 * S0;
  let Sprev = S0;
  for (let i = 1; i <= nSteps; i++) {
    const z = rng.normal();
    const Si = Sprev * Math.exp(drift + diffusion * z);
    S[i] = Si;
    acc += i === nSteps ? 0.5 * Si : Si;
    Sprev = Si;
  }

  return { S, IT: acc * dt };
}

/** Terminal-only sampler: when the caller does not need the full path or I_T,
 * this avoids the Float64Array allocation. Used for spot checks on S_T moments. */
export function sampleTerminal(rng: Rng, opts: SamplePathOpts): number {
  const { S0, mu, sigma, T, nSteps } = opts;
  const dt = T / nSteps;
  const drift = (mu - 0.5 * sigma * sigma) * dt;
  const diffusion = sigma * Math.sqrt(dt);
  let s = S0;
  for (let i = 0; i < nSteps; i++) {
    s *= Math.exp(drift + diffusion * rng.normal());
  }
  return s;
}
