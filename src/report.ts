// Assembles the P&L moments scorecard and break-even table / JSON artifact from a simulation run.

import type { ClosedForm, McResult } from "./core/models.js";
import { closedForm, simulate } from "./core/models.js";
import { expectedHittingTime, firstPassageProb } from "./core/moments.js";
import { simulateSwitching } from "./core/simulate-switching.js";
import type { SwitchingResult } from "./core/simulate-switching.js";
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
  rows: ModelRow[];
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
    partial: number[];
    retained: number[];
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
    barrierRatio: number;
    barrierLevel: number;
    feePost: number;
    probSwitch: { mc: number; closedForm: number | null; zScore: number | null };
    expectedTau: { mc: number; closedForm: number | null; zScore: number | null };
    /** Share of horizon operated in fee mode: E[(T − τ)/T]. */
    meanFracInFeeMode: number;
    /** CVaR₉₅ of Π_{3e} conditional on τ = T (paths that never switched). */
    cvar95GivenNoSwitch: number | null;
    /** CVaR₉₅ of Π_{3e} conditional on τ < T (paths that switched). */
    cvar95GivenSwitch: number | null;
    premiumFair: number;
    premiumLoaded: number;
  };
}

export function makeRow(
  name: string,
  closedMean: number,
  closedSd: number,
  samples: Float64Array,
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

export function subsample(samples: Float64Array, n: number): number[] {
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
  // circuits expensive-looking work to a no-op when h = Infinity (the inner
  // loop still runs but the switching bucket never fires). Skipping the run
  // entirely would desynchronise the RNG tapes between shared-seed runs.
  const switchingRun = simulateSwitching({
    S0: params.S0,
    mu: params.mu,
    sigma: params.sigma,
    P: params.P,
    lambda: params.lambda,
    T: params.T,
    Q: params.Q,
    fee: params.f,
    alpha: params.alpha,
    beta: params.beta,
    premiumLoad: params.premiumLoad,
    premiumMode: params.premiumMode,
    barrierRatio: params.barrierRatio,
    feePost: params.feePost,
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
    makeRow("principal_3b", closed.b2b.mean, closed.b2b.sd, mc.b2b),
    makeRow("principal_3c", closed.partial.mean, closed.partial.sd, mc.partial),
    makeRow("principal_3d", closed.retained.mean, closed.retained.sd, mc.retained),
  ];
  if (isFinite(params.barrierRatio)) {
    // The switching variant has no closed-form moments; NaNs flag this in downstream tables
    // rather than feeding a nonsense z-score to the scorecard.
    rows.push(makeSwitchingRow(switchingRun.pnlSamples));
  }
  // 3a is deterministic: closed-form with zero variance, no MC row.
  rows.unshift({
    name: "principal_3a",
    closedFormMean: closed.matched.mean,
    closedFormSd: 0,
    mcMean: closed.matched.mean,
    mcSd: 0,
    mcCi95: 0,
    zScore: 0,
    var95: -closed.matched.mean,
    var99: -closed.matched.mean,
    cvar95: -closed.matched.mean,
    cvar99: -closed.matched.mean,
    probLoss: closed.matched.mean < 0 ? 1 : 0,
    sharpe: null,
  });

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
      partial: subsample(mc.partial, traceSize),
      retained: subsample(mc.retained, traceSize),
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

// Switching-variant scorecard row: MC-only, no closed-form mean/sd (so NaN-flag those and
// zero out the z-score). VaR/CVaR/probLoss/Sharpe come from the standard
// path-sample closure, identical to every other row.
export function makeSwitchingRow(samples: Float64Array): ModelRow {
  const stats = summarize(samples);
  const sharpe = stats.sd > 0 ? stats.mean / stats.sd : null;
  return {
    name: "principal_3e",
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
  let switched = 0;
  let tauSum = 0;
  let fracFeeSum = 0;
  for (let i = 0; i < nPaths; i++) {
    switched += run.switchedMask[i] as number;
    const tau = run.tauSamples[i] as number;
    tauSum += tau;
    fracFeeSum += (params.T - tau) / params.T;
  }
  const probSwitchMc = switched / nPaths;
  const expectedTauMc = tauSum / nPaths;
  const meanFracInFeeMode = fracFeeSum / nPaths;

  // Closed-form anchors are Harrison/Borodin-Salminen formulas derived under
  // pure GBM. Under Merton jumps the distribution of τ changes (jumps can
  // punch through the barrier), so leave the anchor null and let the report
  // call out "MC only" rather than compare against an inapplicable oracle.
  const pureGbm = params.lambdaJ === 0;
  const probSwitchCf = pureGbm
    ? firstPassageProb(params.mu, params.sigma, params.T, params.barrierRatio)
    : null;
  const expectedTauCf = pureGbm
    ? expectedHittingTime(params.mu, params.sigma, params.T, params.barrierRatio)
    : null;

  // Stderr for a Bernoulli-count and a positive mean — the two anchors z-test
  // on different scales but both use the sample-SD / √nPaths pattern.
  const pSe = Math.sqrt(
    Math.max(1e-12, probSwitchMc * (1 - probSwitchMc)) / nPaths,
  );
  const tauSe = (() => {
    let sse = 0;
    for (let i = 0; i < nPaths; i++) {
      const d = (run.tauSamples[i] as number) - expectedTauMc;
      sse += d * d;
    }
    return nPaths > 1 ? Math.sqrt(sse / (nPaths - 1) / nPaths) : 0;
  })();

  const probZ = probSwitchCf !== null && pSe > 0
    ? (probSwitchMc - probSwitchCf) / pSe
    : null;
  const tauZ = expectedTauCf !== null && tauSe > 0
    ? (expectedTauMc - expectedTauCf) / tauSe
    : null;

  const noSwitch: number[] = [];
  const withSwitch: number[] = [];
  for (let i = 0; i < nPaths; i++) {
    const v = run.pnlSamples[i] as number;
    if (run.switchedMask[i]) withSwitch.push(v);
    else noSwitch.push(v);
  }
  const cvar95GivenNoSwitch =
    noSwitch.length >= 20 ? conditionalVaR(noSwitch, 0.95) : null;
  const cvar95GivenSwitch =
    withSwitch.length >= 20 ? conditionalVaR(withSwitch, 0.95) : null;

  return {
    barrierRatio: params.barrierRatio,
    barrierLevel: run.barrierLevel,
    feePost: run.feePostResolved,
    probSwitch: { mc: probSwitchMc, closedForm: probSwitchCf, zScore: probZ },
    expectedTau: { mc: expectedTauMc, closedForm: expectedTauCf, zScore: tauZ },
    meanFracInFeeMode,
    cvar95GivenNoSwitch,
    cvar95GivenSwitch,
    premiumFair: run.premiumFair,
    premiumLoaded: run.premiumLoaded,
  };
}
