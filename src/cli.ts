import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { buildReport } from "./report.js";
import {
  closedForm,
  partialDeskClosedForm,
  simulate,
} from "./core/models.js";
import type { Params } from "./params.js";
import { defaultParams, withOverrides } from "./params.js";
import { conditionalVaR, probLoss, summarize, valueAtRisk } from "./core/risk.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", "report", "data");

type NumericParamKey = Exclude<keyof Params, "premiumMode">;

const FLAG_TO_PARAM: Record<string, NumericParamKey> = {
  seed: "seed",
  paths: "nPaths",
  steps: "nSteps",
  alpha: "alpha",
  beta: "beta",
  theta: "premiumLoad",
  mu: "mu",
  sigma: "sigma",
  Q: "Q",
  f: "f",
  T: "T",
  lambdaJ: "lambdaJ",
  muJ: "muJ",
  sigmaJ: "sigmaJ",
  h: "barrierRatio",
  fPost: "feePost",
};

interface CliArgs {
  overrides: Partial<Params>;
  sweep: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const overrides: Partial<Params> = {};
  let sweep = false;
  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i];
    if (tok === "--sweep") {
      sweep = true;
      continue;
    }
    if (!tok || !tok.startsWith("--")) continue;
    const key = tok.slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error(`missing value for --${key}`);
    if (key === "premiumMode") {
      if (val !== "sharpe" && val !== "cvar") {
        throw new Error(`--premiumMode must be "sharpe" or "cvar"`);
      }
      overrides.premiumMode = val;
      continue;
    }
    const field = FLAG_TO_PARAM[key];
    if (!field) throw new Error(`unknown flag --${key}`);
    overrides[field] = Number(val);
  }
  return { overrides, sweep };
}

function fmt(x: number, digits = 3): string {
  if (!isFinite(x)) return String(x);
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs >= 1e6 || abs < 1e-3) return x.toExponential(digits);
  return x.toFixed(digits);
}

function printMainTable(params: Params): ReturnType<typeof buildReport> {
  const report = buildReport(params, { keepPaths: 25 });

  console.log(`\nParameters`);
  console.log(
    `  S0=${params.S0}  μ=${params.mu}  σ=${params.sigma}  T=${params.T}` +
      `  P=${params.P}  λ=${params.lambda}  f=${params.f}  Q=${params.Q}` +
      `  α=${params.alpha}  N_paths=${params.nPaths}  N_steps=${params.nSteps}` +
      `  seed=${params.seed}`,
  );
  if (params.lambdaJ > 0) {
    console.log(
      `  jumps: λ_J=${params.lambdaJ}  μ_J=${params.muJ}  σ_J=${params.sigmaJ}` +
        `  (closed-form SD columns remain the GBM anchor)`,
    );
  }

  console.log(`\nOperating books + treasury — closed form vs MC`);
  console.log(
    "  book           E[Π] (cf)     E[Π] (mc)     ±CI95          SD (cf)       SD (mc)       z",
  );
  for (const r of report.rows) {
    console.log(
      `  ${r.name.padEnd(14)}` +
        ` ${fmt(r.closedFormMean).padStart(12)}` +
        ` ${fmt(r.mcMean).padStart(12)}` +
        ` ${("±" + fmt(r.mcCi95)).padStart(14)}` +
        ` ${fmt(r.closedFormSd).padStart(12)}` +
        ` ${fmt(r.mcSd).padStart(12)}` +
        ` ${fmt(r.zScore, 2).padStart(6)}`,
    );
  }

  console.log(`\nTail risk (Monte Carlo) — operating books + treasury`);
  console.log(
    "  book           VaR95         VaR99         CVaR95        CVaR99        P[Π<0]    Sharpe",
  );
  for (const r of report.rows) {
    console.log(
      `  ${r.name.padEnd(14)}` +
        ` ${fmt(r.var95).padStart(12)}` +
        ` ${fmt(r.var99).padStart(12)}` +
        ` ${fmt(r.cvar95).padStart(12)}` +
        ` ${fmt(r.cvar99).padStart(12)}` +
        ` ${fmt(r.probLoss).padStart(8)}` +
        ` ${(r.sharpe === null ? "—" : fmt(r.sharpe, 3)).padStart(8)}`,
    );
  }

  console.log(`\nDesks (operating + treasury) — closed form vs MC`);
  console.log(
    `  matched          E[Π] = ${fmt(report.desks.matched.closedFormMean)}` +
      `  SD = 0 (deterministic, I_T cancels at α = 1)`,
  );
  const pdk = report.desks.partial;
  console.log(
    `  partial (α=${params.alpha})  E[Π] cf=${fmt(pdk.closedFormMean)}` +
      `  mc=${fmt(pdk.mcMean)}  ±CI95=${fmt(pdk.mcCi95)}` +
      `  SD cf=${fmt(pdk.closedFormSd)}  mc=${fmt(pdk.mcSd)}  z=${fmt(pdk.zScore, 2)}`,
  );
  const sdk = report.desks.syndicatedMatched;
  console.log(
    `  syndicated-matched  E[Π] cf=${fmt(sdk.closedFormMean)}` +
      `  mc=${fmt(sdk.mcMean)}  SD mc=${fmt(sdk.mcSd)}`,
  );
  if (report.desks.switchingMatched) {
    const swk = report.desks.switchingMatched;
    console.log(
      `  switching-matched   E[Π] mc=${fmt(swk.mcMean)}  SD mc=${fmt(swk.mcSd)}` +
        `  CVaR95=${fmt(swk.cvar95)}  (MC only)`,
    );
  }

  console.log(
    `\nMatched-inventory NAV shortfall  mean=${fmt(report.drawdown.mean)}` +
      `  sd=${fmt(report.drawdown.sd)}` +
      `  q95=${fmt(report.drawdown.var95)}` +
      `  q99=${fmt(report.drawdown.var99)}` +
      `  max=${fmt(report.drawdown.max)}`,
  );

  if (params.beta > 0 || params.premiumLoad > 0) {
    console.log(
      `\nSyndication  β=${params.beta}  θ=${params.premiumLoad}` +
        `  mode=${params.premiumMode}` +
        `  π_fair=${fmt(report.syndication.premiumFair)}` +
        `  π_loaded=${fmt(report.syndication.premiumLoaded)}`,
    );
  }

  if (report.switching) {
    const sw = report.switching;
    const fmtMaybe = (x: number | null, d = 3) =>
      x === null || !isFinite(x) ? "—" : fmt(x, d);
    console.log(
      `\nSwitching  h=${params.barrierRatio}  H=${fmt(sw.barrierLevel, 4)}` +
        `  f_post=${fmt(sw.feePost, 4)}` +
        (params.feePost === null ? " (locked to f)" : ""),
    );
    console.log(
      `  P[ever in fee]   mc=${fmt(sw.probSwitch.mc, 4)}` +
        `  cf=${fmtMaybe(sw.probSwitch.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.probSwitch.zScore, 2)}`,
    );
    console.log(
      `  E[first-cross∧T] mc=${fmt(sw.expectedCrossingTime.mc, 4)}` +
        `  cf=${fmtMaybe(sw.expectedCrossingTime.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.expectedCrossingTime.zScore, 2)}`,
    );
    console.log(
      `  E[T_fee]         mc=${fmt(sw.meanTimeInFee.mc, 4)}` +
        `  cf=${fmtMaybe(sw.meanTimeInFee.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.meanTimeInFee.zScore, 2)}`,
    );
    console.log(
      `  E[I_fee]         mc=${fmt(sw.meanIntegralFee.mc, 4)}` +
        `  cf=${fmtMaybe(sw.meanIntegralFee.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.meanIntegralFee.zScore, 2)}`,
    );
    console.log(
      `  E[# crossings]=${fmt(sw.meanNCrossings, 3)}` +
        `  CVaR95|no-switch=${fmtMaybe(sw.cvar95GivenNoSwitch)}` +
        `  CVaR95|switched=${fmtMaybe(sw.cvar95GivenSwitch)}`,
    );
  }

  console.log(`\nBreak-even quote  Q* = ${fmt(report.closed.QStar, 4)}`);
  console.log(
    `     E[R_fee] = ${fmt(report.closed.fee.mean)}` +
      `   E[Π_b2b]|Q=Q* = ${fmt(
        report.closed.QStar * report.closed.N -
          params.P * params.lambda * report.closed.IT.mean,
      )}`,
  );
  console.log(
    `\n     Sanity: E[S_T] closed=${fmt(report.terminalSCheck.closedForm)}` +
      `  mc=${fmt(report.terminalSCheck.mcMean)}` +
      `  z=${fmt(report.terminalSCheck.zScore, 2)}`,
  );

  return report;
}

// Sweep grid driving the Validation page's sliders; fewer paths for speed.
const SWEEP_ALPHAS = [0, 0.25, 0.5, 0.75, 1];
const SWEEP_MUS = [-0.1, 0, 0.05, 0.1, 0.2];
const SWEEP_SIGMAS = [0.2, 0.5, 0.8, 1.2];

interface CellMetrics {
  mean: number;
  sd: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  probLoss: number;
  sharpe: number | null;
}

function metricsFromSamples(samples: Float64Array): CellMetrics {
  const s = summarize(samples);
  const sharpe = s.sd > 0 ? s.mean / s.sd : null;
  return {
    mean: s.mean,
    sd: s.sd,
    var95: valueAtRisk(samples, 0.95),
    var99: valueAtRisk(samples, 0.99),
    cvar95: conditionalVaR(samples, 0.95),
    cvar99: conditionalVaR(samples, 0.99),
    probLoss: probLoss(samples),
    sharpe,
  };
}

function runSweep(baseParams: Params): unknown {
  const sweepParams = withOverrides(baseParams, { nPaths: 20_000, nSteps: 100 });
  const cells: unknown[] = [];
  for (const alpha of SWEEP_ALPHAS) {
    for (const mu of SWEEP_MUS) {
      for (const sigma of SWEEP_SIGMAS) {
        const p = withOverrides(sweepParams, { alpha, mu, sigma });
        const cf = closedForm(p);
        const mc = simulate(p);
        const N = p.lambda * p.T;
        const partialDesk = new Float64Array(mc.b2b.length);
        for (let i = 0; i < partialDesk.length; i++) {
          partialDesk[i] = (mc.b2b[i] as number) + (mc.treasury[i] as number);
        }
        const partialDeskCf = partialDeskClosedForm(p);
        const matchedDeskMean = N * (p.Q - p.P * p.S0);
        cells.push({
          alpha,
          mu,
          sigma,
          fee: metricsFromSamples(mc.fee),
          b2b: metricsFromSamples(mc.b2b),
          retained: metricsFromSamples(mc.retained),
          treasury: metricsFromSamples(mc.treasury),
          partialDesk: metricsFromSamples(partialDesk),
          partialDeskClosed: {
            mean: partialDeskCf.mean,
            sd: partialDeskCf.sd,
          },
          matchedDesk: { mean: matchedDeskMean, sd: 0 },
          QStar: cf.QStar,
        });
      }
    }
  }
  return {
    grid: { alphas: SWEEP_ALPHAS, mus: SWEEP_MUS, sigmas: SWEEP_SIGMAS },
    base: sweepParams,
    cells,
  };
}

// Switching-variant threshold sweep: the operator-decision chart (CVaR₉₅
// and E[Π] vs h) is derived from these cells. Infinity = "switch disabled",
// which anchors the curve to the operating-retained reference. Kept
// separate from SWEEP_ALPHAS/MUS/SIGMAS so we don't blow up the (α, μ, σ)
// grid into a 4-dim product.
const SWEEP_BARRIERS = [1.0, 1.1, 1.25, 1.5, 2.0, Infinity];

function runSwitchingSweep(baseParams: Params): unknown {
  const p0 = withOverrides(baseParams, { nPaths: 20_000, nSteps: 100 });
  const cells: unknown[] = [];
  for (const h of SWEEP_BARRIERS) {
    const p = withOverrides(p0, { barrierRatio: h });
    const r = buildReport(p, { keepPaths: 0, traceSize: 0, histBins: 0 });
    const cellMetrics = (name: string): CellMetrics | null => {
      const row = r.rows.find((x) => x.name === name);
      if (!row) return null;
      return {
        mean: row.mcMean,
        sd: row.mcSd,
        var95: row.var95,
        var99: row.var99,
        cvar95: row.cvar95,
        cvar99: row.cvar99,
        probLoss: row.probLoss,
        sharpe: row.sharpe,
      };
    };
    cells.push({
      h,
      switching: r.switching ?? null,
      // `row` is the switching operating book when the threshold is active; at
      // h = Infinity we anchor to the retained operating book so the sweep
      // plot still has a comparable "no-switch" reference.
      row: cellMetrics("switching") ?? cellMetrics("retained"),
      retained: cellMetrics("retained"),
      b2b: cellMetrics("b2b"),
      fee: cellMetrics("fee"),
      treasury: cellMetrics("treasury"),
    });
  }
  return { grid: { barriers: SWEEP_BARRIERS }, base: p0, cells };
}

const QSTAR_MUS = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2];
const QSTAR_TS = [0.25, 0.5, 1, 2, 3];

// Canonical Merton overlay used to verify the compensated-Merton section of
// research-note.md: the compensated drift keeps every closed-form *mean*
// identical to the GBM anchor even with fat, negatively-biased jumps. Fixed
// parameters so the Validation-page verification table is stable across
// reruns.
const JUMP_CHECK: { lambdaJ: number; muJ: number; sigmaJ: number } = {
  lambdaJ: 3,
  muJ: -0.1,
  sigmaJ: 0.15,
};

function runJumpCheck(baseParams: Params): unknown {
  // Rerun with the canonical overlay; reuse the caller's seed/nPaths/nSteps so
  // the MC noise bands match the main-run scale.
  const jumpParams = withOverrides(baseParams, JUMP_CHECK);
  const r = buildReport(jumpParams, { keepPaths: 0, traceSize: 0, histBins: 0 });
  return {
    overlay: JUMP_CHECK,
    rows: r.rows.map((row) => ({
      name: row.name,
      gbmClosedMean: row.closedFormMean,
      gbmClosedSd: row.closedFormSd,
      mertonMcMean: row.mcMean,
      mertonMcCi95: row.mcCi95,
      mertonMcSd: row.mcSd,
      zVsGbmClosed: row.zScore,
    })),
    terminalSCheck: {
      gbmClosed: r.terminalSCheck.closedForm,
      mertonMcMean: r.terminalSCheck.mcMean,
      zVsGbmClosed: r.terminalSCheck.zScore,
    },
  };
}

interface JumpCheckRow {
  name: string;
  gbmClosedMean: number;
  gbmClosedSd: number;
  mertonMcMean: number;
  mertonMcCi95: number;
  mertonMcSd: number;
  zVsGbmClosed: number;
}

interface JumpCheck {
  overlay: { lambdaJ: number; muJ: number; sigmaJ: number };
  rows: JumpCheckRow[];
  terminalSCheck: {
    gbmClosed: number;
    mertonMcMean: number;
    zVsGbmClosed: number;
  };
}

function printJumpCheck(check: unknown): void {
  const c = check as JumpCheck;
  const { lambdaJ, muJ, sigmaJ } = c.overlay;
  console.log(
    `\nCompensated Merton overlay (λ_J=${lambdaJ}, μ_J=${muJ}, σ_J=${sigmaJ})`,
  );
  console.log(
    "  means still match the GBM closed form; SD inflates; see the compensated Merton section of research-note.md",
  );
  console.log(
    "  book           E[Π] gbm-cf   E[Π] merton   ±CI95          SD gbm-cf    SD merton     z",
  );
  for (const r of c.rows) {
    console.log(
      `  ${r.name.padEnd(14)}` +
        ` ${fmt(r.gbmClosedMean).padStart(12)}` +
        ` ${fmt(r.mertonMcMean).padStart(12)}` +
        ` ${("±" + fmt(r.mertonMcCi95)).padStart(14)}` +
        ` ${fmt(r.gbmClosedSd).padStart(12)}` +
        ` ${fmt(r.mertonMcSd).padStart(12)}` +
        ` ${fmt(r.zVsGbmClosed, 2).padStart(6)}`,
    );
  }
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const params = withOverrides(defaultParams, args.overrides);

  mkdirSync(DATA_DIR, { recursive: true });

  const run = printMainTable(params);
  const jumpCheck = runJumpCheck(params);
  printJumpCheck(jumpCheck);
  const runJson = resolve(DATA_DIR, `run-${params.seed}.json`);
  writeFileSync(runJson, JSON.stringify({ ...run, jumpCheck }, null, 2));
  console.log(`\nwrote ${runJson}`);

  if (args.sweep) {
    console.log(`\nRunning parameter sweep…`);
    const sweep = runSweep(params);
    const sweepPath = resolve(DATA_DIR, "sweep.json");
    writeFileSync(sweepPath, JSON.stringify(sweep, null, 2));
    console.log(`wrote ${sweepPath}`);

    console.log(`\nRunning switching-variant barrier sweep…`);
    const switchingSweep = runSwitchingSweep(params);
    const switchingSweepPath = resolve(DATA_DIR, "switching-sweep.json");
    writeFileSync(switchingSweepPath, JSON.stringify(switchingSweep, null, 2));
    console.log(`wrote ${switchingSweepPath}`);
  }

  const qSurface = {
    mus: QSTAR_MUS,
    Ts: QSTAR_TS,
    values: QSTAR_MUS.map((mu) =>
      QSTAR_TS.map((T) => closedForm(withOverrides(params, { mu, T })).QStar),
    ),
  };
  const qPath = resolve(DATA_DIR, "qstar-surface.json");
  writeFileSync(qPath, JSON.stringify(qSurface, null, 2));
  console.log(`wrote ${qPath}`);
}

main();
