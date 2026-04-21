// Assembles the P&L moments scorecard and break-even table / JSON artifact from
// a simulation run. Scorecard rows track the four zero-capital **operating
// books** (fee, b2b, retained, switching) and the **active treasury**; desk
// totals for concrete strategies (matched, partial, syndicated-matched,
// switching-matched, custom) are composed at display time from
// `operating + treasury` sample sums.

import type { ClosedForm, McResult } from "./core/models.js";
import { closedForm, partialDeskClosedForm, simulate } from "./core/models.js";
import {
  expectedHittingTime,
  expectedIntegralAboveBarrier,
  expectedTimeAboveBarrier,
  firstPassageProb,
} from "./core/moments.js";
import { simulateSwitching } from "./core/simulate-switching.js";
import type {
  SwitchingResult,
  SwitchMode,
} from "./core/simulate-switching.js";
import type { Params } from "./params.js";
import {
  conditionalVaR,
  probLoss,
  quantile,
  summarize,
  valueAtRisk,
} from "./core/risk.js";

export interface ModelRow {
  name: string;
  closedFormMean: number;
  closedFormSd: number;
  mcMean: number;
  mcSd: number;
  mcCi95: number;
  /** (mcMean − closedFormMean) / mcStderr. */
  zScore: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  probLoss: number;
  sharpe: number | null;
}

export interface Report {
  params: Params;
  closed: ClosedForm;
  /** Operating books + treasury. Switching row only when `barrierRatio !== Infinity`. */
  rows: ModelRow[];
  /** Closed-form + MC-composed desk totals (operating + treasury at the
   *  strategy's (k_pre, C_basis)). `matched` is deterministic. Switching-desk
   *  rows only when the barrier is active. */
  desks: {
    matched: { closedFormMean: number; closedFormSd: number };
    partial: ModelRow;
    syndicatedMatched: ModelRow;
    switchingMatched?: ModelRow;
  };
  drawdown: {
    mean: number;
    sd: number;
    var95: number;
    var99: number;
    max: number;
  };
  itSampleHistogram: { edges: number[]; counts: number[] };
  terminalSCheck: { mcMean: number; closedForm: number; zScore: number };
  sampledPaths: number[][];
  sampleTraces: {
    fee: number[];
    b2b: number[];
    retained: number[];
    treasury: number[];
    /** Composed `b2b + treasury` at the run's (α·N·P, α·N·P·S_0). */
    partialDesk: number[];
    navDrawdown: number[];
    /** Present only when `params.barrierRatio !== Infinity`. */
    switching?: number[];
  };
  syndication: {
    beta: number;
    premiumLoad: number;
    premiumMode: "sharpe" | "cvar";
    premiumFair: number;
    premiumLoaded: number;
  };
  /** Switching-variant block. Only present when `params.barrierRatio !== Infinity`;
   *  closed-form anchors populated only under pure GBM (lambdaJ = 0). */
  switching?: {
    switchMode: SwitchMode;
    barrierRatio: number;
    barrierLevel: number;
    feePost: number;
    /** P[path ever entered fee mode] = first-passage probability. Same Harrison
     *  oracle under both modes. */
    probSwitch: { mc: number; closedForm: number | null; zScore: number | null };
    /** E[first-passage time ∧ T]. τ is a path property (the first time
     *  S_t ≥ H, or T if never), so the Harrison anchor applies under both
     *  modes — only the book's response to τ differs. CF populated only
     *  under pure GBM (lambdaJ = 0). */
    expectedCrossingTime: { mc: number; closedForm: number | null; zScore: number | null };
    /** E[time spent in fee mode]. Two-way CF via `expectedTimeAboveBarrier`;
     *  one-way CF via `T − expectedHittingTime`. */
    meanTimeInFee: { mc: number; closedForm: number | null; zScore: number | null };
    /** E[I_fee]. Two-way CF via `expectedIntegralAboveBarrier`; one-way leaves
     *  it null (no simple closed form for E[J_τ] under the latched indicator). */
    meanIntegralFee: { mc: number; closedForm: number | null; zScore: number | null };
    /** MC-only, both modes. Under one-way: 0 or 1. */
    meanNCrossings: number;
    /** Share of horizon spent in fee mode: meanTimeInFee.mc / T. */
    meanFracInFeeMode: number;
    /** CVaR₉₅ of switching_op conditional on the path never entering fee mode. */
    cvar95GivenNoSwitch: number | null;
    /** CVaR₉₅ of switching_op conditional on the path entering fee mode at least once. */
    cvar95GivenSwitch: number | null;
  };
}

export function makeRow(
  name: string,
  closedMean: number,
  closedSd: number,
  samples: ArrayLike<number>,
): ModelRow {
  const stats = summarize(samples);
  const sharpe = stats.sd > 0 ? stats.mean / stats.sd : null;
  const zScore = stats.stderr > 0 ? (stats.mean - closedMean) / stats.stderr : 0;
  return {
    name,
    closedFormMean: closedMean,
    closedFormSd: closedSd,
    mcMean: stats.mean,
    mcSd: stats.sd,
    mcCi95: stats.ci95,
    zScore,
    var95: valueAtRisk(samples, 0.95),
    var99: valueAtRisk(samples, 0.99),
    cvar95: conditionalVaR(samples, 0.95),
    cvar99: conditionalVaR(samples, 0.99),
    probLoss: probLoss(samples),
    sharpe,
  };
}

export function histogram(
  samples: Float64Array,
  nBins: number,
): { edges: number[]; counts: number[] } {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i] as number;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!isFinite(lo) || lo === hi) {
    return { edges: [lo, hi], counts: [samples.length] };
  }
  const edges = new Array<number>(nBins + 1);
  for (let i = 0; i <= nBins; i++) edges[i] = lo + ((hi - lo) * i) / nBins;
  const counts = new Array<number>(nBins).fill(0);
  const width = (hi - lo) / nBins;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i] as number;
    let b = Math.floor((v - lo) / width);
    if (b === nBins) b = nBins - 1;
    (counts[b] as number)++;
  }
  return { edges, counts };
}

export function subsample(samples: ArrayLike<number>, n: number): number[] {
  const step = Math.max(1, Math.floor(samples.length / n));
  const out: number[] = [];
  for (let i = 0; i < samples.length && out.length < n; i += step) {
    out.push(samples[i] as number);
  }
  return out;
}

export function buildReport(
  params: Params,
  opts: { keepPaths?: number; traceSize?: number; histBins?: number } = {},
): Report {
  const keepPaths = opts.keepPaths ?? 25;
  const traceSize = opts.traceSize ?? 5_000;
  const histBins = opts.histBins ?? 60;

  const closed = closedForm(params);
  const mc: McResult = simulate(params, { keepPaths });

  // Switching-variant run — shares seed with `simulate(params)` so path-reuse
  // invariance holds (tested explicitly in simulate-switching.test.ts). We
  // always run it even when the barrier is disabled: the wrapper short-
  // circuits expensive-looking work to a no-op when h = Infinity. Skipping
  // the run entirely would desynchronise the RNG tapes between shared-seed
  // runs.
  const switchingRun = simulateSwitching({
    S0: params.S0,
    mu: params.mu,
    sigma: params.sigma,
    P: params.P,
    lambda: params.lambda,
    T: params.T,
    Q: params.Q,
    fee: params.f,
    barrierRatio: params.barrierRatio,
    feePost: params.feePost,
    switchMode: params.switchMode,
    lambdaJ: params.lambdaJ,
    muJ: params.muJ,
    sigmaJ: params.sigmaJ,
    nPaths: params.nPaths,
    nSteps: params.nSteps,
    seed: params.seed,
    keepPaths: 0,
  });

  const rows: ModelRow[] = [
    makeRow("fee", closed.fee.mean, closed.fee.sd, mc.fee),
    makeRow("b2b", closed.b2b.mean, closed.b2b.sd, mc.b2b),
    makeRow("retained", closed.retained.mean, closed.retained.sd, mc.retained),
    makeRow("treasury", closed.treasury.mean, closed.treasury.sd, mc.treasury),
  ];
  if (isFinite(params.barrierRatio)) {
    // The switching operating book has no closed-form moments; NaN-flag those
    // so the scorecard doesn't feed a nonsense z-score.
    rows.push(makeSwitchingRow(switchingRun.pnlSamples));
  }

  // Desk compositions. `matched` is deterministic (I_T cancels between b2b
  // operating and treasury consumption at α = 1); `partial` is the horizon-
  // based composition at params.alpha; syndicated-matched and switching-matched
  // are the operating-layer counterparts composed with the fully-matched
  // treasury.
  const N = closed.N;
  const matchedDeskMean = N * (params.Q - params.P * params.S0);
  const partialDeskSamples = sumSamples(mc.b2b, mc.treasury);
  const matchedTreasury = matchedTreasurySamples(mc, params);
  const syndMatchedSamples = sumSamples(mc.retained, matchedTreasury);
  // At α = 1 the syndicated-matched desk has the matched-deterministic shift
  // plus the syndication cash flow (premium_loaded − β · E[Π_b2b] cancels at
  // θ = 0, is negative otherwise).
  const syndMatchedClosedMean =
    matchedDeskMean + closed.premium.loaded - params.beta * closed.b2b.mean;
  const partialCf = partialDeskClosedForm(params);
  const desks: Report["desks"] = {
    matched: { closedFormMean: matchedDeskMean, closedFormSd: 0 },
    partial: makeRow("partial_desk", partialCf.mean, partialCf.sd, partialDeskSamples),
    syndicatedMatched: makeRow(
      "syndicated_matched_desk",
      syndMatchedClosedMean,
      0,
      syndMatchedSamples,
    ),
    ...(isFinite(params.barrierRatio)
      ? {
          switchingMatched: makeRow(
            "switching_matched_desk",
            NaN,
            NaN,
            sumSamples(switchingRun.pnlSamples, matchedTreasury),
          ),
        }
      : {}),
  };

  const ddStats = summarize(mc.navDrawdowns);
  let ddMax = -Infinity;
  for (let i = 0; i < mc.navDrawdowns.length; i++) {
    const v = mc.navDrawdowns[i] as number;
    if (v > ddMax) ddMax = v;
  }

  const stStats = summarize(mc.terminalS);
  const stClosed = params.S0 * Math.exp(params.mu * params.T);
  const stZ = stStats.stderr > 0 ? (stStats.mean - stClosed) / stStats.stderr : 0;

  const switching = isFinite(params.barrierRatio)
    ? buildSwitchingBlock(params, switchingRun)
    : undefined;

  return {
    params,
    closed,
    rows,
    desks,
    drawdown: {
      mean: ddStats.mean,
      sd: ddStats.sd,
      var95: quantile(mc.navDrawdowns, 0.95),
      var99: quantile(mc.navDrawdowns, 0.99),
      max: ddMax,
    },
    itSampleHistogram: histogram(mc.IT, histBins),
    terminalSCheck: { mcMean: stStats.mean, closedForm: stClosed, zScore: stZ },
    sampledPaths: mc.sampledPaths.map((p) => Array.from(p)),
    sampleTraces: {
      fee: subsample(mc.fee, traceSize),
      b2b: subsample(mc.b2b, traceSize),
      retained: subsample(mc.retained, traceSize),
      treasury: subsample(mc.treasury, traceSize),
      partialDesk: subsample(partialDeskSamples, traceSize),
      navDrawdown: subsample(mc.navDrawdowns, traceSize),
      ...(switching
        ? { switching: subsample(switchingRun.pnlSamples, traceSize) }
        : {}),
    },
    syndication: {
      beta: params.beta,
      premiumLoad: params.premiumLoad,
      premiumMode: params.premiumMode,
      premiumFair: closed.premium.fair,
      premiumLoaded: closed.premium.loaded,
    },
    ...(switching ? { switching } : {}),
  };
}

// Path-by-path `b2b_op + treasury_α` → composed partial-desk samples.
function sumSamples(a: Float64Array, b: Float64Array): Float64Array {
  const n = Math.min(a.length, b.length);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (a[i] as number) + (b[i] as number);
  return out;
}

// Synthetic matched-treasury samples: treasury(k_pre = N·P, C_basis = N·P·S_0)
// = P · λ · I_T − N · P · S_0 (k_left = 0, consumption window = [0, T]).
// Bitwise recovered from the shared I_T tape so I_T cancels path-by-path
// against b2b's − P·λ·I_T term, yielding the matched identity.
function matchedTreasurySamples(mc: McResult, params: Params): Float64Array {
  const N = params.lambda * params.T;
  const shift = N * params.P * params.S0;
  const out = new Float64Array(mc.IT.length);
  for (let i = 0; i < mc.IT.length; i++) {
    out[i] = params.P * params.lambda * (mc.IT[i] as number) - shift;
  }
  return out;
}

// Switching-variant scorecard row: MC-only, no closed-form mean/sd (so NaN-flag
// those and zero out the z-score). VaR/CVaR/probLoss/Sharpe come from the
// standard path-sample closure.
export function makeSwitchingRow(samples: Float64Array): ModelRow {
  const stats = summarize(samples);
  const sharpe = stats.sd > 0 ? stats.mean / stats.sd : null;
  return {
    name: "switching",
    closedFormMean: NaN,
    closedFormSd: NaN,
    mcMean: stats.mean,
    mcSd: stats.sd,
    mcCi95: stats.ci95,
    zScore: 0,
    var95: valueAtRisk(samples, 0.95),
    var99: valueAtRisk(samples, 0.99),
    cvar95: conditionalVaR(samples, 0.95),
    cvar99: conditionalVaR(samples, 0.99),
    probLoss: probLoss(samples),
    sharpe,
  };
}

function buildSwitchingBlock(
  params: Params,
  run: SwitchingResult,
): NonNullable<Report["switching"]> {
  const nPaths = run.pnlSamples.length;
  let everCrossed = 0;
  let tFeeSum = 0;
  let tFeeSumSq = 0;
  let IFeeSum = 0;
  let IFeeSumSq = 0;
  let crossingsSum = 0;
  for (let i = 0; i < nPaths; i++) {
    everCrossed += run.everCrossedMask[i] as number;
    const tFee = run.tFeeSamples[i] as number;
    tFeeSum += tFee;
    tFeeSumSq += tFee * tFee;
    const IFee = run.IFeeSamples[i] as number;
    IFeeSum += IFee;
    IFeeSumSq += IFee * IFee;
    crossingsSum += run.nCrossingsSamples[i] as number;
  }
  const probSwitchMc = everCrossed / nPaths;
  const meanTimeInFeeMc = tFeeSum / nPaths;
  const meanIntegralFeeMc = IFeeSum / nPaths;
  const meanFracInFeeMode = meanTimeInFeeMc / params.T;
  const meanNCrossings = crossingsSum / nPaths;

  // First-crossing time τ is a path property (independent of `switchMode`),
  // so mean(firstCrossTimeSamples) is the direct MC estimate of E[τ ∧ T]
  // and the Harrison / Borodin-Salminen anchor applies under both modes.
  let expectedTauMc = 0;
  for (let i = 0; i < nPaths; i++) {
    expectedTauMc += run.firstCrossTimeSamples[i] as number;
  }
  expectedTauMc /= nPaths;
  let tauSse = 0;
  for (let i = 0; i < nPaths; i++) {
    const d = (run.firstCrossTimeSamples[i] as number) - expectedTauMc;
    tauSse += d * d;
  }
  const expectedTauSe =
    nPaths > 1 ? Math.sqrt(tauSse / (nPaths - 1) / nPaths) : 0;

  // Closed-form anchors hold under pure GBM. Under Merton jumps the τ / I_fee
  // distributions change (jumps can punch through the barrier), so leave
  // those anchors null and let the report call out "MC only" rather than
  // compare against an inapplicable oracle.
  const pureGbm = params.lambdaJ === 0;
  const probSwitchCf = pureGbm
    ? firstPassageProb(params.mu, params.sigma, params.T, params.barrierRatio)
    : null;
  const expectedTauCf = pureGbm
    ? expectedHittingTime(params.mu, params.sigma, params.T, params.barrierRatio)
    : null;

  // E[T_fee] closed form. Under two-way: ∫₀ᵀ Φ((νt − log h)/(σ√t)) dt — the
  // fee-mode indicator tracks S_t across H symmetrically so re-entries into
  // b2b mode subtract from T_fee. Under one-way the book latches at τ, so
  // E[T_fee] = T − E[τ ∧ T] exactly.
  const meanTimeInFeeCf = pureGbm
    ? run.switchMode === "two-way"
      ? expectedTimeAboveBarrier(
          params.mu,
          params.sigma,
          params.T,
          params.barrierRatio,
        )
      : expectedTauCf !== null
        ? params.T - expectedTauCf
        : null
    : null;

  // E[I_fee] closed form. Only available under two-way + pure GBM.
  const meanIntegralFeeCf =
    pureGbm && run.switchMode === "two-way"
      ? expectedIntegralAboveBarrier(
          params.S0,
          params.mu,
          params.sigma,
          params.T,
          params.barrierRatio,
        )
      : null;

  const pSe = Math.sqrt(
    Math.max(1e-12, probSwitchMc * (1 - probSwitchMc)) / nPaths,
  );
  const probZ = probSwitchCf !== null && pSe > 0
    ? (probSwitchMc - probSwitchCf) / pSe
    : null;
  const tauZ = expectedTauCf !== null && expectedTauSe > 0
    ? (expectedTauMc - expectedTauCf) / expectedTauSe
    : null;

  const tFeeVar = Math.max(0, tFeeSumSq / nPaths - meanTimeInFeeMc * meanTimeInFeeMc);
  const tFeeSe = Math.sqrt(tFeeVar / nPaths);
  const tFeeZ = meanTimeInFeeCf !== null && tFeeSe > 0
    ? (meanTimeInFeeMc - meanTimeInFeeCf) / tFeeSe
    : null;

  const IFeeVar = Math.max(0, IFeeSumSq / nPaths - meanIntegralFeeMc * meanIntegralFeeMc);
  const IFeeSe = Math.sqrt(IFeeVar / nPaths);
  const IFeeZ = meanIntegralFeeCf !== null && IFeeSe > 0
    ? (meanIntegralFeeMc - meanIntegralFeeCf) / IFeeSe
    : null;

  const noSwitch: number[] = [];
  const withSwitch: number[] = [];
  for (let i = 0; i < nPaths; i++) {
    const v = run.pnlSamples[i] as number;
    if (run.everCrossedMask[i]) withSwitch.push(v);
    else noSwitch.push(v);
  }
  const cvar95GivenNoSwitch =
    noSwitch.length >= 20 ? conditionalVaR(noSwitch, 0.95) : null;
  const cvar95GivenSwitch =
    withSwitch.length >= 20 ? conditionalVaR(withSwitch, 0.95) : null;

  return {
    switchMode: run.switchMode,
    barrierRatio: params.barrierRatio,
    barrierLevel: run.barrierLevel,
    feePost: run.feePostResolved,
    probSwitch: { mc: probSwitchMc, closedForm: probSwitchCf, zScore: probZ },
    expectedCrossingTime: {
      mc: expectedTauMc,
      closedForm: expectedTauCf,
      zScore: tauZ,
    },
    meanTimeInFee: {
      mc: meanTimeInFeeMc,
      closedForm: meanTimeInFeeCf,
      zScore: tFeeZ,
    },
    meanIntegralFee: {
      mc: meanIntegralFeeMc,
      closedForm: meanIntegralFeeCf,
      zScore: IFeeZ,
    },
    meanNCrossings,
    meanFracInFeeMode,
    cvar95GivenNoSwitch,
    cvar95GivenSwitch,
  };
}
