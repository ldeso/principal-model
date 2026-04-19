// Assembles the §4 / §5 report table and JSON artifact from a simulation run.

import type { ClosedForm, McResult } from "./core/models.js";
import { closedForm, simulate } from "./core/models.js";
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
  };
  syndication: {
    beta: number;
    premiumLoad: number;
    premiumMode: "sharpe" | "cvar";
    premiumFair: number;
    premiumLoaded: number;
  };
}

function makeRow(
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

function histogram(
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

function subsample(samples: Float64Array, n: number): number[] {
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

  const rows: ModelRow[] = [
    makeRow("fee", closed.fee.mean, closed.fee.sd, mc.fee),
    makeRow("principal_3b", closed.b2b.mean, closed.b2b.sd, mc.b2b),
    makeRow("principal_3c", closed.partial.mean, closed.partial.sd, mc.partial),
    makeRow("principal_3d", closed.retained.mean, closed.retained.sd, mc.retained),
  ];
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
    },
    syndication: {
      beta: params.beta,
      premiumLoad: params.premiumLoad,
      premiumMode: params.premiumMode,
      premiumFair: closed.premium.fair,
      premiumLoaded: closed.premium.loaded,
    },
  };
}
