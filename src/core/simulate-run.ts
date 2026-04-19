// Phase C custom-principal Monte Carlo runner. Shares the Merton path core
// with src/core/gbm.ts; adds the fee / b2b / custom-principal accumulators
// that used to be hand-ported inline in report/lib/ojs-helpers.js. Both the
// OJS cells and the CLI can call this so the browser and the offline report
// run the same code.

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
  /** §3d external cession fraction of the residual stochastic leg. Default 0. */
  beta?: number;
  /** §3d counterparty risk-load multiplier θ ≥ 0. Default 0 ⇒ fair premium. */
  premiumLoad?: number;
  /** §3d risk-measure basis for the load. Default `"sharpe"`. */
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
  feeSamples: Float64Array;
  principalSamples: Float64Array;
  b2bSamples: Float64Array;
  /** §3d retained P&L after quota-share syndication of the residual leg. */
  retainedSamples: Float64Array;
  /** §3d loaded premium applied to `retainedSamples` (MC-moment derived). */
  premium: number;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** Coverage fraction τ_cov of the horizon that the pre-purchased inventory
   *  funds, clamped to [0, 1]. Distinct from §3e's stopping time τ (see
   *  `./simulate-switching.ts`); the two quantities share a letter in the
   *  research note but never in code. */
  tauFrac: number;
  tokensUsedInternal: number;
  tokensLeftover: number;
  /** Inventory notional N = λ · T. */
  N: number;
}

// §3d Gaussian CVaR95 factor = φ(Φ^{-1}(0.95))/0.05; proxy used in `"cvar"` mode.
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
  // Snap tauFrac (τ_cov) onto the integration grid and treat "only the endpoint
  // falls in [τ_cov·T, T]" as an empty range — otherwise the trapezoid rule adds
  // 0.5·S_T·dt for a zero-length interval.
  const tailStartRaw = Math.ceil(tauFrac * nSteps);
  const tailStartStep = tailStartRaw >= nSteps ? nSteps + 1 : tailStartRaw;

  const feeSamples = new Float64Array(nPaths);
  const principalSamples = new Float64Array(nPaths);
  const b2bSamples = new Float64Array(nPaths);
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

    // Uncovered-tail integral ∫_{τ_cov·T}^{T} S_t dt, trapezoid on the same grid
    // so the same realisation drives fee, b2b and custom-principal books.
    let tailInt = 0;
    for (let k = tailStartStep; k <= nSteps; k++) {
      const w = (k === tailStartStep || k === nSteps) ? 0.5 : 1;
      tailInt += w * (S[k] as number);
    }
    tailInt *= dt;

    terminalS[i] = ST;
    ITSamples[i] = IT;
    feeSamples[i] = fee * P * lambda * IT;
    b2bSamples[i] = Q * N - P * lambda * IT;
    principalSamples[i] =
      Q * N - cBasis - P * lambda * tailInt + tokensLeftover * ST;

    if (i < keep) sampledPaths.push(S);
  }

  // §3d quota-share on the residual stochastic leg. Phase B uses a
  // closed-form premium because Π_α is linear in I_T; Phase C's custom
  // inventory has no clean closed-form counterpart (the matched slice mixes
  // τ_cov with a sunk basis), so we price the cession from this run's own
  // MC sample moments. With nPaths ≥ 5000 the estimator is tight enough for
  // the slider feedback loop. `cBasis` is the deterministic translation of
  // the principal book, so the stochastic residual is principal − (Q·N − cBasis).
  const detShift = Q * N - cBasis;
  let stochMean = 0;
  for (let i = 0; i < nPaths; i++) {
    stochMean += (principalSamples[i] as number) - detShift;
  }
  stochMean /= nPaths;
  let stochVar = 0;
  for (let i = 0; i < nPaths; i++) {
    const d = (principalSamples[i] as number) - detShift - stochMean;
    stochVar += d * d;
  }
  stochVar = nPaths > 1 ? stochVar / (nPaths - 1) : 0;
  const stochSd = Math.sqrt(stochVar);
  const loadFactor = premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const piFair = beta * stochMean;
  const premium = piFair - beta * premiumLoad * loadFactor * stochSd;

  const retainedSamples = new Float64Array(nPaths);
  for (let i = 0; i < nPaths; i++) {
    retainedSamples[i] =
      detShift + (1 - beta) * ((principalSamples[i] as number) - detShift) + premium;
  }

  return {
    feeSamples,
    principalSamples,
    b2bSamples,
    retainedSamples,
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
