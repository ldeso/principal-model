import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { buildReport } from "./report.js";
import { closedForm } from "./core/models.js";
import type { Params } from "./params.js";
import { defaultParams, withOverrides } from "./params.js";

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

  console.log(`\nP&L moments, closed-form vs MC`);
  console.log(
    "  model          E[Π] (cf)     E[Π] (mc)     ±CI95          SD (cf)       SD (mc)       z",
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

  console.log(`\nTail risk (Monte Carlo)`);
  console.log(
    "  model          VaR95         VaR99         CVaR95        CVaR99        P[Π<0]    Sharpe",
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

  console.log(
    `\nMatched book — NAV drawdown  mean=${fmt(report.drawdown.mean)}` +
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
      `  P[τ<T]  mc=${fmt(sw.probSwitch.mc, 4)}` +
        `  cf=${fmtMaybe(sw.probSwitch.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.probSwitch.zScore, 2)}`,
    );
    console.log(
      `  E[τ∧T]  mc=${fmt(sw.expectedTau.mc, 4)}` +
        `  cf=${fmtMaybe(sw.expectedTau.closedForm, 4)}` +
        `  z=${fmtMaybe(sw.expectedTau.zScore, 2)}`,
    );
    console.log(
      `  E[(T−τ)/T]=${fmt(sw.meanFracInFeeMode, 4)}` +
        `  CVaR95|no-switch=${fmtMaybe(sw.cvar95GivenNoSwitch)}` +
        `  CVaR95|switched=${fmtMaybe(sw.cvar95GivenSwitch)}`,
    );
    console.log(
      `  π_fair=${fmt(sw.premiumFair)}  π_loaded=${fmt(sw.premiumLoaded)}` +
        `  (MC-derived; no closed form for Π_sw)`,
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

// Sweep grid driving the Observable report's sliders; fewer paths for speed.
const SWEEP_ALPHAS = [0, 0.25, 0.5, 0.75, 1];
const SWEEP_MUS = [-0.1, 0, 0.05, 0.1, 0.2];
const SWEEP_SIGMAS = [0.2, 0.5, 0.8, 1.2];

function runSweep(baseParams: Params): unknown {
  const sweepParams = withOverrides(baseParams, { nPaths: 20_000, nSteps: 100 });
  const cells: unknown[] = [];
  for (const alpha of SWEEP_ALPHAS) {
    for (const mu of SWEEP_MUS) {
      for (const sigma of SWEEP_SIGMAS) {
        const p = withOverrides(sweepParams, { alpha, mu, sigma });
        const r = buildReport(p, { keepPaths: 0, traceSize: 0, histBins: 40 });
        cells.push({
          alpha,
          mu,
          sigma,
          fee: extractRowMetrics(r.rows, "fee"),
          b2b: extractRowMetrics(r.rows, "principal_3b"),
          matched: extractRowMetrics(r.rows, "principal_3a"),
          partial: extractRowMetrics(r.rows, "principal_3c"),
          drawdown: r.drawdown,
          QStar: r.closed.QStar,
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

function extractRowMetrics(
  rows: ReturnType<typeof buildReport>["rows"],
  name: string,
): unknown {
  const r = rows.find((x) => x.name === name);
  if (!r) return null;
  return {
    mean: r.mcMean,
    sd: r.mcSd,
    var95: r.var95,
    var99: r.var99,
    cvar95: r.cvar95,
    cvar99: r.cvar99,
    probLoss: r.probLoss,
    sharpe: r.sharpe,
  };
}

// Switching-variant barrier sweep: the operator-decision chart (CVaR₉₅ and E[Π] vs h) is
// derived from these cells. Infinity = "switch disabled", which anchors the
// curve to the syndicated retained book. Kept separate from SWEEP_ALPHAS/MUS/SIGMAS
// so we don't blow up the (α, μ, σ) grid into a 4-dim product.
const SWEEP_BARRIERS = [1.0, 1.1, 1.25, 1.5, 2.0, Infinity];

function runSwitchingSweep(baseParams: Params): unknown {
  const p0 = withOverrides(baseParams, { nPaths: 20_000, nSteps: 100 });
  const cells: unknown[] = [];
  for (const h of SWEEP_BARRIERS) {
    const p = withOverrides(p0, { barrierRatio: h });
    const r = buildReport(p, { keepPaths: 0, traceSize: 0, histBins: 0 });
    const switchingRow =
      r.rows.find((x) => x.name === "principal_3e") ?? null;
    cells.push({
      h,
      switching: r.switching ?? null,
      row: switchingRow
        ? extractRowMetrics(r.rows, "principal_3e")
        : extractRowMetrics(r.rows, "principal_3d"),
      retained: extractRowMetrics(r.rows, "principal_3d"),
      b2b: extractRowMetrics(r.rows, "principal_3b"),
      fee: extractRowMetrics(r.rows, "fee"),
    });
  }
  return { grid: { barriers: SWEEP_BARRIERS }, base: p0, cells };
}

const QSTAR_MUS = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2];
const QSTAR_TS = [0.25, 0.5, 1, 2, 3];

// Canonical Merton overlay used to verify the compensated Merton section of research-note.md:
// the compensated
// drift keeps every closed-form *mean* identical to the GBM anchor even with
// fat, negatively-biased jumps. Fixed parameters so the Validation-page
// verification table is stable across reruns.
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
    "  model          E[Π] gbm-cf   E[Π] merton   ±CI95          SD gbm-cf    SD merton     z",
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
