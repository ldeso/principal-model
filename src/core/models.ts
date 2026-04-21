// Closed-form moments and MC sampler for the operating books (fee, b2b,
// retained) and the active treasury. Desk totals are `operating + treasury`
// sample-wise; at α = 1 the kernel cancels and the matched desk reduces to
// N·(Q − P·S_0).

import { samplePath } from "./gbm.js";
import {
  covarITST,
  expectedIt,
  expectedST,
  expm1OverX,
  secondMomentIt,
  varianceIt,
  varianceST,
} from "./moments.js";
import type { Params } from "../params.js";
import { mulberry32 } from "./rng.js";
import { shortfallVsSchedule } from "./risk.js";

export interface ClosedForm {
  /** Fee: f·P·λ·I_T. */
  fee: { mean: number; variance: number; sd: number };
  /** B2b: Q·N − P·λ·I_T. */
  b2b: { mean: number; variance: number; sd: number };
  /** Retained: (1 − β)·Π_b2b + π_syn. */
  retained: { mean: number; variance: number; sd: number };
  /** Treasury at (α·N·P, α·N·P·S_0). */
  treasury: { mean: number; variance: number; sd: number };
  /** Syndication premium scalars. */
  premium: { fair: number; loaded: number };
  /** Break-even quote. */
  QStar: number;
  IT: { mean: number; variance: number };
  /** N = λ·T. */
  N: number;
}

// φ(Φ⁻¹(0.95)) / 0.05; CVaR-mode load factor.
const GAUSSIAN_CVAR95_FACTOR = 2.062713055949736;

// Closed forms hold under pure GBM. Under compensated Merton the jump
// compensation preserves E[S_t], so E[I_τ], E[S_T], and every mean-level
// identity carry over; variances do not.
export function closedForm(p: Params): ClosedForm {
  const N = p.lambda * p.T;
  const { mean: eIt, variance: vIt } = gbmITMoments(p);

  const feeMean = p.f * p.P * p.lambda * eIt;
  const feeVar = (p.f * p.P * p.lambda) ** 2 * vIt;

  const b2bMean = p.Q * N - p.P * p.lambda * eIt;
  const b2bVar = (p.P * p.lambda) ** 2 * vIt;

  const b2bSd = Math.sqrt(b2bVar);
  const loadFactor = p.premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const piFair = p.beta * b2bMean;
  const piLoaded = piFair - p.beta * p.premiumLoad * loadFactor * b2bSd;
  const retainedMean = (1 - p.beta) * b2bMean + piLoaded;
  const retainedVar = (1 - p.beta) ** 2 * b2bVar;

  // Treasury at (α·N·P, α·N·P·S_0). Over-hedged branch covers α > 1.
  const kPre = p.alpha * N * p.P;
  const cBasis = p.alpha * N * p.P * p.S0;
  const kLeft = Math.max(0, kPre - N * p.P);
  const tau = Math.min(p.T, p.alpha * p.T);
  const consumptionMean = expectedIt(p.S0, p.mu, tau);
  const consumptionVar = varianceIt(p.S0, p.mu, p.sigma, tau);
  const treasuryMean =
    p.P * p.lambda * consumptionMean + kLeft * expectedST(p.S0, p.mu, p.T) -
    cBasis;
  const treasuryVar = kLeft === 0
    ? (p.P * p.lambda) ** 2 * consumptionVar
    : (p.P * p.lambda) ** 2 * vIt +
      kLeft * kLeft * varianceST(p.S0, p.mu, p.sigma, p.T) +
      2 * p.P * p.lambda * kLeft * covarITST(p.S0, p.mu, p.sigma, p.T);

  const QStar = (1 + p.f) * p.P * p.S0 * expm1OverX(p.mu * p.T);

  return {
    fee: { mean: feeMean, variance: feeVar, sd: Math.sqrt(feeVar) },
    b2b: { mean: b2bMean, variance: b2bVar, sd: b2bSd },
    retained: {
      mean: retainedMean,
      variance: retainedVar,
      sd: Math.sqrt(retainedVar),
    },
    treasury: {
      mean: treasuryMean,
      variance: treasuryVar,
      sd: Math.sqrt(Math.max(0, treasuryVar)),
    },
    premium: { fair: piFair, loaded: piLoaded },
    QStar,
    IT: { mean: eIt, variance: vIt },
    N,
  };
}

function gbmITMoments(p: Params): { mean: number; variance: number } {
  return {
    mean: expectedIt(p.S0, p.mu, p.T),
    variance: varianceIt(p.S0, p.mu, p.sigma, p.T),
  };
}

// Partial-desk closed form via J_α = S_{αT}·Y with Y an independent
// unit-start GBM integral over [0, (1−α)T].
export interface DeskClosedForm {
  mean: number;
  variance: number;
  sd: number;
}

export function partialDeskClosedForm(p: Params): DeskClosedForm {
  const N = p.lambda * p.T;
  const alpha = Math.max(0, Math.min(1, p.alpha));
  const tau = alpha * p.T;
  const tail = (1 - alpha) * p.T;
  const eStauOverS0 = Math.exp(p.mu * tau);
  const eY = tail * expm1OverX(p.mu * tail);
  const eJalpha = p.S0 * eStauOverS0 * eY;
  const mean = p.Q * N - alpha * N * p.P * p.S0 - p.P * p.lambda * eJalpha;

  const eS2tau = p.S0 * p.S0 * Math.exp((2 * p.mu + p.sigma * p.sigma) * tau);
  const eY2 = secondMomentIt(1, p.mu, p.sigma, tail);
  const eJalphaMean = p.S0 * eStauOverS0 * eY;
  const varJalpha = Math.max(0, eS2tau * eY2 - eJalphaMean * eJalphaMean);
  const variance = (p.P * p.lambda) ** 2 * varJalpha;
  return { mean, variance, sd: Math.sqrt(variance) };
}

export interface McSamples {
  fee: Float64Array;
  b2b: Float64Array;
  retained: Float64Array;
  treasury: Float64Array;
  /** Loaded syndication premium (scalar). */
  premium: number;
  IT: Float64Array;
  /** max_t shortfall-vs-schedule on matched inventory. */
  navDrawdowns: Float64Array;
  terminalS: Float64Array;
}

export interface SampleOpts {
  /** Full paths retained for plotting; capped by nPaths. */
  keepPaths?: number;
}

export interface McResult extends McSamples {
  sampledPaths: Float64Array[];
}

export function simulate(p: Params, opts: SampleOpts = {}): McResult {
  const cf = closedForm(p);
  const rng = mulberry32(p.seed);
  const N = p.lambda * p.T;

  const fee = new Float64Array(p.nPaths);
  const b2b = new Float64Array(p.nPaths);
  const retained = new Float64Array(p.nPaths);
  const treasury = new Float64Array(p.nPaths);
  const IT = new Float64Array(p.nPaths);
  const navDrawdowns = new Float64Array(p.nPaths);
  const terminalS = new Float64Array(p.nPaths);

  const keep = Math.min(opts.keepPaths ?? 0, p.nPaths);
  const sampledPaths: Float64Array[] = [];

  const dt = p.T / p.nSteps;
  const alpha = p.alpha;
  const kPre = alpha * N * p.P;
  const cBasis = alpha * N * p.P * p.S0;
  const kLeft = Math.max(0, kPre - N * p.P);
  const navNotional = N * p.P;

  // Snap τ_cov onto the grid; on-grid boundary ⇒ no zero-width trapezoid.
  const alphaFrac = Math.min(1, alpha);
  const tailStartRaw = Math.ceil(alphaFrac * p.nSteps);
  const tailStartStep = tailStartRaw >= p.nSteps ? p.nSteps + 1 : tailStartRaw;

  const premium = cf.premium.loaded;

  for (let i = 0; i < p.nPaths; i++) {
    const path = samplePath(rng, {
      S0: p.S0,
      mu: p.mu,
      sigma: p.sigma,
      T: p.T,
      nSteps: p.nSteps,
      lambdaJ: p.lambdaJ,
      muJ: p.muJ,
      sigmaJ: p.sigmaJ,
    });
    IT[i] = path.IT;
    terminalS[i] = path.S[p.nSteps] as number;

    let tailInt = 0;
    for (let k = tailStartStep; k <= p.nSteps; k++) {
      const w = k === tailStartStep || k === p.nSteps ? 0.5 : 1;
      tailInt += w * (path.S[k] as number);
    }
    tailInt *= dt;
    const consumptionInt = path.IT - tailInt;

    fee[i] = p.f * p.P * p.lambda * path.IT;
    const b2bVal = p.Q * N - p.P * p.lambda * path.IT;
    b2b[i] = b2bVal;
    retained[i] = (1 - p.beta) * b2bVal + premium;
    treasury[i] =
      p.P * p.lambda * consumptionInt + kLeft * (path.S[p.nSteps] as number) -
      cBasis;

    navDrawdowns[i] = shortfallVsSchedule(path.S, navNotional);

    if (i < keep) sampledPaths.push(path.S);
  }

  return {
    fee,
    b2b,
    retained,
    treasury,
    premium,
    IT,
    navDrawdowns,
    terminalS,
    sampledPaths,
  };
}
