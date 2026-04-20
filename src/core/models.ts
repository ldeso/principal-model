// Per-model P&L closed-form moments and Monte Carlo sampler. See the
// fee-based, principal, and direct-comparison sections of research-note.md.

import { samplePath } from "./gbm.js";
import { gbmMoments, expm1OverX } from "./moments.js";
import type { Params } from "../params.js";
import { mulberry32 } from "./rng.js";
import { shortfallVsSchedule } from "./risk.js";

export interface ClosedForm {
  fee: { mean: number; variance: number; sd: number };
  matched: { mean: number; variance: 0; sd: 0 };
  b2b: { mean: number; variance: number; sd: number };
  partial: { mean: number; variance: number; sd: number };
  /** Syndicated-variant retained P&L after ceding β of the (1−α) stochastic leg for π. */
  retained: { mean: number; variance: number; sd: number };
  /** Syndicated-variant premium scalars. `fair = β(1−α)·E[Π_b2b]`; `loaded` deducts
   *  θ·(risk-measure of the ceded leg) and is what `simulate()` applies. */
  premium: { fair: number; loaded: number };
  /** Break-even quote equalising E[Π_b2b] with E[R_fee]. */
  QStar: number;
  IT: { mean: number; variance: number };
  /** Inventory notional N = λ · T. */
  N: number;
}

// Gaussian CVaR95 shape factor = φ(Φ^{-1}(0.95)) / (1 − 0.95) ≈ 2.062713.
// First-pass proxy for E[−(Q·N − P·λ·I_T) | tail] / SD[Π_b2b]; the true
// I_T tail is non-Gaussian so MC remains authoritative (see the syndicated-variant section of research-note.md).
const GAUSSIAN_CVAR95_FACTOR = 2.062713055949736;

// Closed-form moments are derived under pure GBM. Under compensated Merton
// jump-diffusion (λ_J > 0) E[I_T] is invariant — the compensation cancels
// jumps' mean effect — so fee/b2b/partial *means* and Q* are still exact.
// Var[I_T] does change under jumps; the variance figures below remain the
// GBM anchor, and the MC path (src/gbm.ts samplePath) is authoritative for
// the true jump-aware variance and tail metrics.
export function closedForm(p: Params): ClosedForm {
  const N = p.lambda * p.T;
  const { mean: eIt, variance: vIt } = gbmMoments(p.S0, p.mu, p.sigma, p.T);

  const feeMean = p.f * p.P * p.lambda * eIt;
  const feeVar = (p.f * p.P * p.lambda) ** 2 * vIt;

  const matchedMean = N * (p.Q - p.P * p.S0);

  const b2bMean = p.Q * N - p.P * p.lambda * eIt;
  const b2bVar = (p.P * p.lambda) ** 2 * vIt;

  const partialMean = (1 - p.alpha) * b2bMean + p.alpha * matchedMean;
  const partialVar = (1 - p.alpha) ** 2 * b2bVar;

  // Syndicated-variant quota-share on the (1−α) stochastic leg only — the matched slice is
  // deterministic, syndicating it is vacuous. Premium is a closed-form scalar
  // so the retained-variance identity stays exact under MC.
  const b2bSd = Math.sqrt(b2bVar);
  const loadFactor =
    p.premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const cession = p.beta * (1 - p.alpha);
  const piFair = cession * b2bMean;
  const piLoaded = piFair - cession * p.premiumLoad * loadFactor * b2bSd;
  const retainedMean =
    p.alpha * matchedMean + (1 - p.alpha) * (1 - p.beta) * b2bMean + piLoaded;
  const retainedVar = ((1 - p.alpha) * (1 - p.beta)) ** 2 * b2bVar;

  // Q* = (1 + f) · P · S_0 · (e^{μT} − 1)/(μT); expm1OverX handles μ → 0.
  const QStar = (1 + p.f) * p.P * p.S0 * expm1OverX(p.mu * p.T);

  return {
    fee: { mean: feeMean, variance: feeVar, sd: Math.sqrt(feeVar) },
    matched: { mean: matchedMean, variance: 0, sd: 0 },
    b2b: { mean: b2bMean, variance: b2bVar, sd: b2bSd },
    partial: {
      mean: partialMean,
      variance: partialVar,
      sd: Math.sqrt(partialVar),
    },
    retained: {
      mean: retainedMean,
      variance: retainedVar,
      sd: Math.sqrt(retainedVar),
    },
    premium: { fair: piFair, loaded: piLoaded },
    QStar,
    IT: { mean: eIt, variance: vIt },
    N,
  };
}

export interface McSamples {
  fee: Float64Array;
  b2b: Float64Array;
  partial: Float64Array;
  /** Syndicated-variant retained P&L after quota-share syndication of the stochastic leg. */
  retained: Float64Array;
  /** Syndicated-variant loaded premium actually applied to `retained` (closed-form scalar). */
  premium: number;
  /** Deterministic; returned scalar for table symmetry. */
  matched: number;
  /** Shared random kernel I_T. */
  IT: Float64Array;
  /** max_{t ≤ T} shortfall-vs-schedule per path (3a). */
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
  const matchedPnL = N * (p.Q - p.P * p.S0);

  const fee = new Float64Array(p.nPaths);
  const b2b = new Float64Array(p.nPaths);
  const partial = new Float64Array(p.nPaths);
  const retained = new Float64Array(p.nPaths);
  const IT = new Float64Array(p.nPaths);
  const navDrawdowns = new Float64Array(p.nPaths);
  const terminalS = new Float64Array(p.nPaths);

  const keep = Math.min(opts.keepPaths ?? 0, p.nPaths);
  const sampledPaths: Float64Array[] = [];

  const alphaShift = p.alpha * p.S0 * p.T;
  const navNotional = N * p.P;
  const retainedStochScale = (1 - p.alpha) * (1 - p.beta);
  const retainedMatched = p.alpha * matchedPnL;
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

    fee[i] = p.f * p.P * p.lambda * path.IT;
    const b2bPath = p.Q * N - p.P * p.lambda * path.IT;
    b2b[i] = b2bPath;
    partial[i] =
      p.Q * N - p.P * p.lambda * (alphaShift + (1 - p.alpha) * path.IT);
    retained[i] = retainedMatched + retainedStochScale * b2bPath + premium;

    navDrawdowns[i] = shortfallVsSchedule(path.S, navNotional);

    if (i < keep) sampledPaths.push(path.S);
  }

  return {
    fee,
    b2b,
    partial,
    retained,
    premium,
    matched: matchedPnL,
    IT,
    navDrawdowns,
    terminalS,
    sampledPaths,
  };
}
