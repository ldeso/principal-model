// Operating-book + treasury Monte Carlo runner. Shares the Merton path core
// with src/core/gbm.ts; emits the fee, b2b, retained (syndicated-on-b2b)
// operating samples plus the active-treasury samples parameterised by
// (kPre, cBasis). Desk totals for concrete strategies are `operating + treasury`
// evaluated sample-wise by the consumer.

import { samplePath } from "./gbm.js";
import { mulberry32 } from "./rng.js";

export interface SimulateRunInputs {
  S0: number;
  mu: number;
  sigma: number;
  /** Protocol price kVCM/tonne. */
  P: number;
  /** Retirement flow, tonnes per unit time. */
  lambda: number;
  /** Horizon in the same time unit as μ, σ, λ. */
  T: number;
  /** Fixed USD quote per tonne. */
  Q: number;
  /** Fee rate. */
  fee: number;
  /** Pre-purchased inventory (tonnes) available at t = 0. */
  kPre: number;
  /** Sunk cost basis of the pre-purchase. */
  cBasis: number;
  /** Syndicated-variant external cession fraction of the b2b book. Default 0. */
  beta?: number;
  /** Syndicated-variant counterparty risk-load multiplier θ ≥ 0. Default 0 ⇒ fair premium. */
  premiumLoad?: number;
  /** Syndicated-variant risk-measure basis for the load. Default `"sharpe"`. */
  premiumMode?: "sharpe" | "cvar";
  /** Merton jump intensity. 0 ⇒ pure GBM. */
  lambdaJ?: number;
  /** Mean of log-jump size. */
  muJ?: number;
  /** SD of log-jump size. */
  sigmaJ?: number;
  nPaths: number;
  nSteps: number;
  seed: number;
  /** Sample trajectories retained for plotting; capped by nPaths. */
  keepPaths?: number;
}

export interface SimulateRunResult {
  /** Fee operating book: f · P · λ · I_T. */
  feeSamples: Float64Array;
  /** Back-to-back operating book: Q · N − P · λ · I_T. */
  b2bSamples: Float64Array;
  /** Syndicated-on-b2b operating book: (1 − β) · b2b + premium. */
  retainedSamples: Float64Array;
  /** Active treasury at (kPre, cBasis):
   *  P · λ · I_{[0, min(T, kPre/(P·λ))]} + max(0, kPre − N·P) · S_T − cBasis. */
  treasurySamples: Float64Array;
  /** Loaded syndication premium applied to `retainedSamples` (MC-moment derived). */
  premium: number;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** Coverage fraction τ_cov of the horizon that the pre-purchased inventory
   *  funds, clamped to [0, 1]. Distinct from the switching variant's stopping
   *  time τ (see `./simulate-switching.ts`). */
  tauFrac: number;
  tokensUsedInternal: number;
  tokensLeftover: number;
  /** Inventory notional N = λ · T. */
  N: number;
}

// Gaussian CVaR95 factor = φ(Φ^{-1}(0.95))/0.05; proxy used in `"cvar"` mode.
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
  // Snap tauFrac onto the integration grid and treat "only the endpoint falls
  // in [τ_cov·T, T]" as an empty range — otherwise the trapezoid rule adds
  // 0.5·S_T·dt for a zero-length interval.
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

    // Uncovered-tail integral ∫_{τ_cov·T}^{T} S_t dt on the same grid so the
    // same realisation drives every book.
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

  // Syndicated premium — closed-form in principle (β · E[Π_b2b]), but we derive
  // it from this run's b2b sample moments so the OJS slider feedback loop
  // matches the MC cross-check path-by-path (premium is applied as a scalar).
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
