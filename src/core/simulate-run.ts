// Operating-book + treasury MC runner. Emits fee, b2b, retained, and
// active-treasury samples at (kPre, cBasis). Shares the Merton kernel with
// src/core/gbm.ts; desks are assembled by the consumer.

import { samplePath } from "./gbm.js";
import { mulberry32 } from "./rng.js";

export interface SimulateRunInputs {
  S0: number;
  mu: number;
  sigma: number;
  P: number;
  lambda: number;
  T: number;
  Q: number;
  fee: number;
  kPre: number;
  cBasis: number;
  /** Default 0. */
  beta?: number;
  /** θ ≥ 0; default 0. */
  premiumLoad?: number;
  /** Default `sharpe`. */
  premiumMode?: "sharpe" | "cvar";
  /** Default 0 ⇒ pure GBM. */
  lambdaJ?: number;
  muJ?: number;
  sigmaJ?: number;
  nPaths: number;
  nSteps: number;
  seed: number;
  /** Sample paths retained; capped by nPaths. */
  keepPaths?: number;
}

export interface SimulateRunResult {
  feeSamples: Float64Array;
  b2bSamples: Float64Array;
  retainedSamples: Float64Array;
  /** Treasury at (kPre, cBasis). */
  treasurySamples: Float64Array;
  /** Loaded premium (scalar). */
  premium: number;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** Coverage fraction τ_cov / T ∈ [0, 1]. */
  tauFrac: number;
  tokensUsedInternal: number;
  tokensLeftover: number;
  /** N = λ·T. */
  N: number;
}

const GAUSSIAN_CVAR95_FACTOR = 2.062713055949736;

export function simulateRun(inputs: SimulateRunInputs): SimulateRunResult {
  const {
    S0, mu, sigma, P, lambda, T, Q, fee, kPre, cBasis,
    beta = 0, premiumLoad = 0, premiumMode = "sharpe",
    lambdaJ = 0, muJ = 0, sigmaJ = 0,
    nPaths, nSteps, seed,
    keepPaths = 0,
  } = inputs;

  const rng = mulberry32(seed);
  const dt = T / nSteps;
  const N = lambda * T;
  const tauFrac = (lambda > 0 && P > 0)
    ? Math.min(1, kPre / (lambda * P * T))
    : 1;
  const tokensUsedInternal = Math.min(kPre, lambda * T * P);
  const tokensLeftover = Math.max(0, kPre - lambda * T * P);
  // Snap τ_cov onto the grid; on-grid boundary ⇒ no zero-width trapezoid.
  const tailStartRaw = Math.ceil(tauFrac * nSteps);
  const tailStartStep = tailStartRaw >= nSteps ? nSteps + 1 : tailStartRaw;

  const feeSamples = new Float64Array(nPaths);
  const b2bSamples = new Float64Array(nPaths);
  const treasurySamples = new Float64Array(nPaths);
  const ITSamples = new Float64Array(nPaths);
  const terminalS = new Float64Array(nPaths);
  const keep = Math.min(keepPaths, nPaths);
  const sampledPaths: Float64Array[] = [];

  for (let i = 0; i < nPaths; i++) {
    const path = samplePath(rng, {
      S0, mu, sigma, T, nSteps, lambdaJ, muJ, sigmaJ,
    });
    const S = path.S;
    const IT = path.IT;
    const ST = S[nSteps] as number;

    let tailInt = 0;
    for (let k = tailStartStep; k <= nSteps; k++) {
      const w = (k === tailStartStep || k === nSteps) ? 0.5 : 1;
      tailInt += w * (S[k] as number);
    }
    tailInt *= dt;
    const consumptionInt = IT - tailInt;

    terminalS[i] = ST;
    ITSamples[i] = IT;
    feeSamples[i] = fee * P * lambda * IT;
    b2bSamples[i] = Q * N - P * lambda * IT;
    treasurySamples[i] =
      P * lambda * consumptionInt + tokensLeftover * ST - cBasis;

    if (i < keep) sampledPaths.push(S);
  }

  // Premium from this run's b2b sample moments so OJS slider feedback
  // matches path-wise (premium is applied as a scalar).
  let b2bMean = 0;
  for (let i = 0; i < nPaths; i++) b2bMean += b2bSamples[i] as number;
  b2bMean /= nPaths;
  let b2bSse = 0;
  for (let i = 0; i < nPaths; i++) {
    const d = (b2bSamples[i] as number) - b2bMean;
    b2bSse += d * d;
  }
  const b2bVar = nPaths > 1 ? b2bSse / (nPaths - 1) : 0;
  const b2bSd = Math.sqrt(b2bVar);
  const loadFactor = premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const piFair = beta * b2bMean;
  const premium = piFair - beta * premiumLoad * loadFactor * b2bSd;

  const retainedSamples = new Float64Array(nPaths);
  for (let i = 0; i < nPaths; i++) {
    retainedSamples[i] = (1 - beta) * (b2bSamples[i] as number) + premium;
  }

  return {
    feeSamples,
    b2bSamples,
    retainedSamples,
    treasurySamples,
    premium,
    ITSamples,
    terminalS,
    sampledPaths,
    tauFrac,
    tokensUsedInternal,
    tokensLeftover,
    N,
  };
}
