import { describe, expect, it } from "vitest";
import { mulberry32 } from "../src/core/rng.js";
import { samplePath } from "../src/core/gbm.js";
import {
  expectedHittingTime,
  firstPassageProb,
  standardNormalCdf,
} from "../src/core/moments.js";
import { conditionalVaR, summarize } from "../src/core/risk.js";
import { simulate } from "../src/core/models.js";
import { simulateSwitching } from "../src/core/simulate-switching.js";
import type { SwitchingInputs } from "../src/core/simulate-switching.js";
import { defaultParams, withOverrides } from "../src/params.js";

// Baseline inputs: pure GBM (λ_J = 0) keeps the first-passage CDF exact;
// moderate nPaths for fast iteration.
const base: SwitchingInputs = {
  S0: 1,
  mu: 0.05,
  sigma: 0.3,
  P: 1,
  lambda: 1_000,
  T: 1,
  Q: 1.1,
  fee: 0.05,
  barrierRatio: Infinity,
  feePost: null,
  nPaths: 2_000,
  nSteps: 200,
  seed: 2026,
};

describe("simulateSwitching — switching operating book", () => {
  it("barrierRatio = Infinity makes every path unswitched (τ = T)", () => {
    const r = simulateSwitching(base);
    for (let i = 0; i < r.switchedMask.length; i++) {
      expect(r.switchedMask[i]).toBe(0);
      expect(r.tauSamples[i]).toBe(base.T);
    }
    // With no switch, J_τ = 0 and the operating book collapses onto b2b.
    for (let i = 0; i < r.pnlSamples.length; i++) {
      expect(r.pnlSamples[i]).toBeCloseTo(r.b2bSamples[i] as number, 10);
    }
  });

  it("barrierRatio ≤ 1 fires the switch at t = 0 and the book = fee · P · λ · I_T", () => {
    const r = simulateSwitching({ ...base, barrierRatio: 1 });
    for (let i = 0; i < r.tauSamples.length; i++) {
      expect(r.tauSamples[i]).toBe(0);
      expect(r.switchedMask[i]).toBe(1);
      expect(r.ITauSamples[i]).toBe(0);
    }
    for (let i = 0; i < r.pnlSamples.length; i++) {
      expect(r.pnlSamples[i]).toBeCloseTo(r.feeSamples[i] as number, 10);
    }
  });

  it("hand-computes switching_op = Q·λ·τ − P·λ·I_τ + f_post·P·λ·J_τ", () => {
    const p = {
      ...base,
      barrierRatio: 1.2,
      nPaths: 30,
      nSteps: 40,
      seed: 9999,
    };
    const r = simulateSwitching(p);
    const fPost = p.feePost ?? p.fee;
    for (let i = 0; i < r.pnlSamples.length; i++) {
      const tau = r.tauSamples[i] as number;
      const ITau = r.ITauSamples[i] as number;
      const JTau = r.JTauSamples[i] as number;
      const expected =
        p.Q * p.lambda * tau -
        p.P * p.lambda * ITau +
        fPost * p.P * p.lambda * JTau;
      expect(r.pnlSamples[i]).toBeCloseTo(expected, 9);
      expect(ITau + JTau).toBeCloseTo(r.ITSamples[i] as number, 12);
    }
    const r2 = simulateSwitching({ ...p, feePost: 0.15 });
    for (let i = 0; i < r2.pnlSamples.length; i++) {
      const tau = r2.tauSamples[i] as number;
      const ITau = r2.ITauSamples[i] as number;
      const JTau = r2.JTauSamples[i] as number;
      const expected =
        p.Q * p.lambda * tau -
        p.P * p.lambda * ITau +
        0.15 * p.P * p.lambda * JTau;
      expect(r2.pnlSamples[i]).toBeCloseTo(expected, 9);
    }
  });

  it("feePost = 0 reduces the book to Q·λ·τ − P·λ·I_τ per path", () => {
    const p = { ...base, barrierRatio: 1.2, feePost: 0 };
    const r = simulateSwitching(p);
    for (let i = 0; i < r.pnlSamples.length; i++) {
      const tau = r.tauSamples[i] as number;
      const ITau = r.ITauSamples[i] as number;
      const expected = p.Q * p.lambda * tau - p.P * p.lambda * ITau;
      expect(r.pnlSamples[i]).toBeCloseTo(expected, 9);
    }
  });

  it("feePost = f equals feePost = null path-by-path (lock-to-f)", () => {
    const shared = { ...base, barrierRatio: 1.25 };
    const rNull = simulateSwitching({ ...shared, feePost: null });
    const rLocked = simulateSwitching({ ...shared, feePost: shared.fee });
    for (let i = 0; i < rNull.pnlSamples.length; i++) {
      expect(rNull.pnlSamples[i]).toBeCloseTo(rLocked.pnlSamples[i] as number, 10);
    }
    expect(rNull.feePostResolved).toBe(shared.fee);
    expect(rLocked.feePostResolved).toBe(shared.fee);
  });

  it("shares RNG with models.simulate under shared seed (I_T parity path-by-path)", () => {
    const p = withOverrides(defaultParams, {
      nPaths: 64,
      nSteps: 50,
      seed: 1234,
      lambdaJ: 0,
      muJ: 0,
      sigmaJ: 0,
      barrierRatio: Infinity,
    });
    const mc = simulate(p);
    const sw = simulateSwitching({
      S0: p.S0, mu: p.mu, sigma: p.sigma, P: p.P, lambda: p.lambda, T: p.T,
      Q: p.Q, fee: p.f,
      barrierRatio: p.barrierRatio, feePost: p.feePost,
      lambdaJ: 0, muJ: 0, sigmaJ: 0,
      nPaths: p.nPaths, nSteps: p.nSteps, seed: p.seed,
    });
    for (let i = 0; i < p.nPaths; i++) {
      expect(sw.ITSamples[i]).toBeCloseTo(mc.IT[i] as number, 12);
      expect(sw.terminalS[i]).toBeCloseTo(mc.terminalS[i] as number, 12);
    }
  });

  it("shares RNG with samplePath under shared seed (bit-for-bit terminal prices)", () => {
    const p = {
      ...base,
      barrierRatio: Infinity,
      lambdaJ: 0,
      nPaths: 16,
      nSteps: 32,
    };
    const r = simulateSwitching(p);
    const rng = mulberry32(p.seed);
    for (let i = 0; i < p.nPaths; i++) {
      const path = samplePath(rng, {
        S0: p.S0, mu: p.mu, sigma: p.sigma, T: p.T, nSteps: p.nSteps,
      });
      expect(r.terminalS[i]).toBe(path.S[p.nSteps]);
      expect(r.ITSamples[i]).toBeCloseTo(path.IT, 12);
    }
  });

  it("MC P[τ ≤ T] matches firstPassageProb under pure GBM within 4·stderr", () => {
    const cases = [
      { h: 1.2, mu: 0.05, sigma: 0.3, T: 1 },
      { h: 1.5, mu: 0.0, sigma: 0.5, T: 1 },
      { h: 2.0, mu: 0.1, sigma: 0.4, T: 2 },
    ];
    for (const c of cases) {
      const p = {
        ...base,
        mu: c.mu, sigma: c.sigma, T: c.T, barrierRatio: c.h,
        nPaths: 20_000, nSteps: 1_000,
      };
      const r = simulateSwitching(p);
      let hits = 0;
      for (let i = 0; i < r.switchedMask.length; i++) hits += r.switchedMask[i] as number;
      const pMc = hits / r.switchedMask.length;
      const pCf = firstPassageProb(c.mu, c.sigma, c.T, c.h);
      const se = Math.sqrt((pMc * (1 - pMc)) / r.switchedMask.length);
      expect(Math.abs(pMc - pCf)).toBeLessThan(4 * se + 0.01);
    }
  });

  it("MC E[τ∧T] matches expectedHittingTime within 4·stderr (pure GBM)", () => {
    const cases = [
      { h: 1.2, mu: 0.05, sigma: 0.3, T: 1 },
      { h: 1.5, mu: 0.0, sigma: 0.5, T: 1.5 },
    ];
    for (const c of cases) {
      const p = {
        ...base,
        mu: c.mu, sigma: c.sigma, T: c.T, barrierRatio: c.h,
        nPaths: 20_000, nSteps: 1_000,
      };
      const r = simulateSwitching(p);
      const s = summarize(r.tauSamples);
      const cfT = expectedHittingTime(c.mu, c.sigma, c.T, c.h);
      expect(Math.abs(s.mean - cfT)).toBeLessThan(4 * s.stderr + 0.01);
    }
  });

  it("discrete-crossing bias decreases monotonically in nSteps", () => {
    const base1 = {
      ...base,
      mu: 0.0, sigma: 0.4, barrierRatio: 1.3, nPaths: 30_000, seed: 9001,
    };
    const pCf = firstPassageProb(base1.mu, base1.sigma, base1.T, base1.barrierRatio);
    const biases: number[] = [];
    for (const nSteps of [50, 250, 1_000]) {
      const r = simulateSwitching({ ...base1, nSteps });
      let hits = 0;
      for (let i = 0; i < r.switchedMask.length; i++) hits += r.switchedMask[i] as number;
      const pMc = hits / r.switchedMask.length;
      biases.push(Math.abs(pMc - pCf));
    }
    expect(biases[1]).toBeLessThan(biases[0] as number);
    expect(biases[2]).toBeLessThan(biases[1] as number);
  });

  it("compensated Merton jumps strictly raise P[τ ≤ T] at equal σ", () => {
    const shared = {
      ...base,
      mu: 0.0, sigma: 0.3, barrierRatio: 1.4,
      nPaths: 30_000, nSteps: 500, seed: 4242,
    };
    const gbm = simulateSwitching({ ...shared, lambdaJ: 0, muJ: 0, sigmaJ: 0 });
    const jumps = simulateSwitching({
      ...shared, lambdaJ: 15, muJ: 0, sigmaJ: 0.2,
    });
    const pGbm = count1s(gbm.switchedMask) / gbm.switchedMask.length;
    const pJumps = count1s(jumps.switchedMask) / jumps.switchedMask.length;
    expect(pJumps).toBeGreaterThan(pGbm);
  });

  it("CVaR decomposition brackets the overall CVaR by the switched-mask partition", () => {
    const p = {
      ...base,
      barrierRatio: 1.25,
      nPaths: 20_000, nSteps: 500, seed: 8888,
    };
    const r = simulateSwitching(p);
    const noSwitch: number[] = [];
    const withSwitch: number[] = [];
    for (let i = 0; i < r.pnlSamples.length; i++) {
      const v = r.pnlSamples[i] as number;
      if (r.switchedMask[i]) withSwitch.push(v);
      else noSwitch.push(v);
    }
    expect(noSwitch.length).toBeGreaterThan(100);
    expect(withSwitch.length).toBeGreaterThan(100);
    const overall = conditionalVaR(r.pnlSamples, 0.95);
    const cNo = conditionalVaR(noSwitch, 0.95);
    const cSw = conditionalVaR(withSwitch, 0.95);
    const lo = Math.min(cNo, cSw);
    const hi = Math.max(cNo, cSw);
    const slack = 0.02 * Math.max(Math.abs(cNo), Math.abs(cSw), 1);
    expect(overall).toBeGreaterThan(lo - slack);
    expect(overall).toBeLessThan(hi + slack);
  });
});

describe("moments.ts — first-passage helpers", () => {
  it("h ≤ 1 returns P = 1 and E[τ] = 0 (barrier already breached)", () => {
    expect(firstPassageProb(0.05, 0.3, 1, 1)).toBe(1);
    expect(firstPassageProb(0.05, 0.3, 1, 0.9)).toBe(1);
    expect(expectedHittingTime(0.05, 0.3, 1, 1)).toBe(0);
  });

  it("firstPassageProb is monotone decreasing in h for fixed (μ, σ, T)", () => {
    const ps = [1.05, 1.1, 1.25, 1.5, 2.0].map((h) =>
      firstPassageProb(0.05, 0.3, 1, h),
    );
    for (let i = 1; i < ps.length; i++) {
      expect(ps[i]).toBeLessThan(ps[i - 1] as number);
    }
  });

  it("standardNormalCdf matches known anchor values", () => {
    expect(standardNormalCdf(0)).toBeCloseTo(0.5, 7);
    expect(standardNormalCdf(1.96)).toBeCloseTo(0.9750021048517795, 6);
    expect(standardNormalCdf(-2)).toBeCloseTo(0.022750131948179195, 6);
  });

  it("expectedHittingTime ≤ T always", () => {
    for (const h of [1.2, 1.5, 2, 5]) {
      const e = expectedHittingTime(0.05, 0.3, 1, h);
      expect(e).toBeLessThanOrEqual(1);
      expect(e).toBeGreaterThanOrEqual(0);
    }
  });
});

function count1s(mask: Uint8Array): number {
  let k = 0;
  for (let i = 0; i < mask.length; i++) k += mask[i] as number;
  return k;
}
