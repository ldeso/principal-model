// Per-book P&L closed-form moments and Monte Carlo sampler. The intermediary's
// books split cleanly into zero-capital **operating** books (fee, b2b, retained,
// switching — each with P&L(0) = 0, every tonne sourced at spot) and a single
// balance-sheet **treasury** parameterised by initial tokens `k_pre` and sunk
// basis `C_basis`. Desk totals for composite strategies are `operating + treasury`
// evaluated path-by-path; closed-form desk moments follow by the J_α algebra in
// the accompanying research note.
//
// At params.alpha the treasury carries notional α·N·P tokens at basis α·N·P·S_0,
// so the composed desk `b2b_op + treasury` reproduces the horizon-based partial
// coverage. At α = 1 the integral I_T cancels exactly between `b2b_op` and the
// treasury's consumption of the full inventory, yielding the deterministic
// matched desk N·(Q − P·S_0).

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
  /** Fee operating book: f·P·λ·I_T. */
  fee: { mean: number; variance: number; sd: number };
  /** Back-to-back operating book: Q·N − P·λ·I_T. */
  b2b: { mean: number; variance: number; sd: number };
  /** Syndicated-on-b2b operating book: (1 − β)·Π_b2b + π_syn. */
  retained: { mean: number; variance: number; sd: number };
  /** Active treasury at (k_pre = α·N·P, C_basis = α·N·P·S_0). */
  treasury: { mean: number; variance: number; sd: number };
  /** Syndication premium scalars on the (unsliced) operating book. */
  premium: { fair: number; loaded: number };
  /** Break-even quote equalising E[R_fee] and E[Π_b2b]. */
  QStar: number;
  IT: { mean: number; variance: number };
  /** Inventory notional N = λ · T. */
  N: number;
}

// Gaussian CVaR95 shape factor = φ(Φ^{-1}(0.95)) / (1 − 0.95) ≈ 2.062713.
// `cvar`-mode load rescales the Sharpe-mode θ·SD[Π_b2b] onto a CVaR scale; the
// true I_T tail is heavier than Gaussian, so MC remains authoritative for tail
// metrics.
const GAUSSIAN_CVAR95_FACTOR = 2.062713055949736;

// Closed forms are derived under pure GBM. Under compensated Merton
// jump-diffusion (λ_J > 0) the jump compensation preserves E[S_t], so
// E[I_τ] and E[S_T] — and therefore E[treasury], E[Π_b2b], E[R_fee], Q* —
// carry over unchanged. Variances do not; the MC path (src/gbm.ts
// samplePath) remains authoritative for jump-aware variance and tails.
export function closedForm(p: Params): ClosedForm {
  const N = p.lambda * p.T;
  const { mean: eIt, variance: vIt } = gbmITMoments(p);

  // Fee and back-to-back operating books (unchanged from the note).
  const feeMean = p.f * p.P * p.lambda * eIt;
  const feeVar = (p.f * p.P * p.lambda) ** 2 * vIt;

  const b2bMean = p.Q * N - p.P * p.lambda * eIt;
  const b2bVar = (p.P * p.lambda) ** 2 * vIt;

  // Syndicated-on-b2b operating book. Premium is a closed-form scalar so the
  // variance identity is exact under MC. Cession applies to the full operating
  // book — the matched slice no longer lives inside `retained_op`, it composes
  // in at the treasury/desk level.
  const b2bSd = Math.sqrt(b2bVar);
  const loadFactor = p.premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const piFair = p.beta * b2bMean;
  const piLoaded = piFair - p.beta * p.premiumLoad * loadFactor * b2bSd;
  const retainedMean = (1 - p.beta) * b2bMean + piLoaded;
  const retainedVar = (1 - p.beta) ** 2 * b2bVar;

  // Treasury at (k_pre, C_basis) = (α·N·P, α·N·P·S_0). Under/exactly hedged
  // when α ≤ 1 (k_left = 0); over-hedged branch covers α > 1 (the UI ordinarily
  // clamps at 1 but `Params.alpha` admits larger values, so we handle it).
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

  // Q* = (1 + f) · P · S_0 · (e^{μT} − 1)/(μT); expm1OverX handles μ → 0.
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

// Small wrapper so the closed-form and simulate paths share the same pure-GBM
// anchor moments even after the file shed `gbmMoments` in favour of per-purpose
// helpers.
function gbmITMoments(p: Params): { mean: number; variance: number } {
  return {
    mean: expectedIt(p.S0, p.mu, p.T),
    variance: varianceIt(p.S0, p.mu, p.sigma, p.T),
  };
}

// Closed-form desk total for `b2b_op + treasury(α·N·P, α·N·P·S_0)`. Useful for
// the scorecard's "matched desk" and "partial desk" summary rows (the
// composition is MC-exact path-by-path, but the desk moments have a clean
// Dufresne-style closed form too, derived from the strong Markov
// decomposition J_α = S_{αT} · Y' with Y' an independent unit-start GBM
// integral over [0, (1−α)T]).
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
  // E[J_α] = E[S_τ] · E[Y] with Y = ∫₀^{(1−α)T} S'_s ds, S'_0 = 1.
  const eStauOverS0 = Math.exp(p.mu * tau);
  const eY = tail * expm1OverX(p.mu * tail);
  const eJalpha = p.S0 * eStauOverS0 * eY;
  const mean = p.Q * N - alpha * N * p.P * p.S0 - p.P * p.lambda * eJalpha;

  // Var[J_α] = E[S_τ²]·E[Y²] − (E[S_τ]·E[Y])².
  const eS2tau = p.S0 * p.S0 * Math.exp((2 * p.mu + p.sigma * p.sigma) * tau);
  const eY2 = secondMomentIt(1, p.mu, p.sigma, tail);
  const eJalphaMean = p.S0 * eStauOverS0 * eY;
  const varJalpha = Math.max(0, eS2tau * eY2 - eJalphaMean * eJalphaMean);
  const variance = (p.P * p.lambda) ** 2 * varJalpha;
  return { mean, variance, sd: Math.sqrt(variance) };
}

export interface McSamples {
  /** Fee operating book. */
  fee: Float64Array;
  /** Back-to-back operating book. */
  b2b: Float64Array;
  /** Syndicated-on-b2b operating book (pure; matched slice lives in treasury). */
  retained: Float64Array;
  /** Active treasury at (α·N·P, α·N·P·S_0). */
  treasury: Float64Array;
  /** Loaded syndication premium applied to `retained` (closed-form scalar). */
  premium: number;
  /** Shared random kernel I_T. */
  IT: Float64Array;
  /** max_{t ≤ T} shortfall-vs-schedule of the matched-inventory NAV decay. */
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

  // Snap the inventory-exhaustion time onto the integration grid. Same
  // convention as simulate-run.ts: "the step that completes the coverage
  // window" belongs to the consumption bucket so an exactly-on-grid boundary
  // doesn't produce a zero-width trapezoid.
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

    // Tail integral ∫_{αT}^{T} S_t dt (same trapezoid split as simulate-run.ts)
    // so the consumption integral is I_T − tailInt.
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
