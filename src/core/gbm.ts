// Log-exact GBM sampler with trapezoidal I_T. Compensated Merton overlay;
// λ_J = 0 reproduces pure GBM bit-for-bit.

import type { Rng } from "./rng.js";

export interface GbmPath {
  /** Prices at t_0 = 0, …, t_N = T. Length nSteps + 1. */
  S: Float64Array;
  /** Trapezoidal ∫₀ᵀ S_t dt. */
  IT: number;
}

export interface SamplePathOpts {
  S0: number;
  mu: number;
  sigma: number;
  T: number;
  nSteps: number;
  /** Merton intensity (/yr). Default 0. */
  lambdaJ?: number;
  /** Mean log-jump. Default 0. */
  muJ?: number;
  /** Log-jump SD. Default 0. */
  sigmaJ?: number;
}

export function samplePath(rng: Rng, opts: SamplePathOpts): GbmPath {
  const { S0, mu, sigma, T, nSteps } = opts;
  const lambdaJ = opts.lambdaJ ?? 0;
  const muJ = opts.muJ ?? 0;
  const sigmaJ = opts.sigmaJ ?? 0;

  const dt = T / nSteps;
  const kappa = lambdaJ > 0
    ? Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1
    : 0;
  const drift = (mu - 0.5 * sigma * sigma - lambdaJ * kappa) * dt;
  const diffusion = sigma * Math.sqrt(dt);
  const lamDt = lambdaJ * dt;

  const S = new Float64Array(nSteps + 1);
  S[0] = S0;

  let acc = 0.5 * S0;
  let Sprev = S0;
  for (let i = 1; i <= nSteps; i++) {
    const z = rng.normal();
    let jumpSum = 0;
    if (lambdaJ > 0) {
      const n = rng.poisson(lamDt);
      for (let k = 0; k < n; k++) {
        jumpSum += muJ + sigmaJ * rng.normal();
      }
    }
    const Si = Sprev * Math.exp(drift + diffusion * z + jumpSum);
    S[i] = Si;
    acc += i === nSteps ? 0.5 * Si : Si;
    Sprev = Si;
  }

  return { S, IT: acc * dt };
}
