import { describe, expect, it } from "vitest";
import { samplePath } from "../src/core/gbm.js";
import { expectedIt } from "../src/core/moments.js";
import { mulberry32 } from "../src/core/rng.js";
import { summarize } from "../src/core/risk.js";
import { simulateRun } from "../src/core/simulate-run.js";
import type { SimulateRunInputs } from "../src/core/simulate-run.js";

const base: SimulateRunInputs = {
  S0: 1,
  mu: 0.05,
  sigma: 0.3,
  P: 1,
  lambda: 1000,
  T: 1,
  Q: 1.1,
  fee: 0.05,
  kPre: 0,
  cBasis: 0,
  nPaths: 1_000,
  nSteps: 50,
  seed: 2026,
};

describe("simulateRun: active treasury", () => {
  it("kPre = 0, cBasis = 0 ⇒ treasury ≡ 0", () => {
    const r = simulateRun({ ...base, kPre: 0, cBasis: 0 });
    expect(r.tauFrac).toBe(0);
    expect(r.tokensLeftover).toBe(0);
    for (let i = 0; i < r.treasurySamples.length; i++) {
      expect(r.treasurySamples[i]).toBe(0);
    }
  });

  it("kPre = 0, cBasis > 0 ⇒ treasury ≡ −cBasis", () => {
    const cBasis = 12;
    const r = simulateRun({ ...base, kPre: 0, cBasis });
    for (let i = 0; i < r.treasurySamples.length; i++) {
      expect(r.treasurySamples[i]).toBe(-cBasis);
    }
  });

  it("kPre = λ·P·T ⇒ treasury = P·λ·I_T − cBasis", () => {
    const p = { ...base, kPre: base.lambda * base.P * base.T, cBasis: 40 };
    const r = simulateRun(p);
    expect(r.tauFrac).toBe(1);
    expect(r.tokensLeftover).toBe(0);
    expect(r.tokensUsedInternal).toBe(p.kPre);
    for (let i = 0; i < r.treasurySamples.length; i++) {
      const expected = p.P * p.lambda * (r.ITSamples[i] as number) - p.cBasis;
      expect(r.treasurySamples[i]).toBeCloseTo(expected, 8);
    }
  });

  it("kPre > λ·P·T ⇒ leftover marked at S_T", () => {
    const covered = base.lambda * base.P * base.T;
    const p = { ...base, kPre: 2 * covered, cBasis: 77 };
    const r = simulateRun(p);
    expect(r.tauFrac).toBe(1);
    expect(r.tokensLeftover).toBe(covered);
    expect(r.tokensUsedInternal).toBe(covered);
    for (let i = 0; i < r.treasurySamples.length; i++) {
      const expected =
        p.P * p.lambda * (r.ITSamples[i] as number) +
        r.tokensLeftover * (r.terminalS[i] as number) -
        p.cBasis;
      expect(r.treasurySamples[i]).toBeCloseTo(expected, 8);
    }
  });

  it("matched desk: b2b + treasury(N·P, N·P·S_0) = N·(Q − P·S_0)", () => {
    const covered = base.lambda * base.P * base.T;
    const p = { ...base, kPre: covered, cBasis: covered * base.S0 };
    const r = simulateRun(p);
    const expected = p.lambda * p.T * (p.Q - p.P * p.S0);
    for (let i = 0; i < r.b2bSamples.length; i++) {
      const desk = (r.b2bSamples[i] as number) + (r.treasurySamples[i] as number);
      expect(desk).toBeCloseTo(expected, 8);
    }
  });

  it("E[R_fee] matches GBM closed form", () => {
    const p = { ...base, nPaths: 20_000, nSteps: 100 };
    const r = simulateRun(p);
    const eIT = expectedIt(p.S0, p.mu, p.T);
    const expectedFeeMean = p.fee * p.P * p.lambda * eIT;
    const s = summarize(r.feeSamples);
    expect(Math.abs(s.mean - expectedFeeMean)).toBeLessThan(4 * s.stderr);
  });

  it("shares the Merton kernel with samplePath under a common seed", () => {
    const p = { ...base, nPaths: 16, nSteps: 32 };
    const r = simulateRun(p);
    const rng = mulberry32(p.seed);
    for (let i = 0; i < p.nPaths; i++) {
      const path = samplePath(rng, {
        S0: p.S0, mu: p.mu, sigma: p.sigma, T: p.T, nSteps: p.nSteps,
      });
      expect(r.ITSamples[i]).toBe(path.IT);
      expect(r.terminalS[i]).toBe(path.S[p.nSteps]);
    }
  });

  it("tauFrac = 1 when λ or P is zero", () => {
    const rZero = simulateRun({ ...base, lambda: 0, kPre: 0, cBasis: 0 });
    expect(rZero.tauFrac).toBe(1);
    expect(rZero.tokensUsedInternal).toBe(0);
    expect(rZero.tokensLeftover).toBe(0);
  });
});

describe("simulateRun: syndicated-on-b2b", () => {
  it("β = 0 ⇒ retained ≡ b2b, premium 0", () => {
    const r = simulateRun({ ...base, kPre: 50_000, cBasis: 4_000, beta: 0 });
    expect(r.premium).toBe(0);
    for (let i = 0; i < r.b2bSamples.length; i++) {
      expect(r.retainedSamples[i]).toBeCloseTo(r.b2bSamples[i] as number, 12);
    }
  });

  it("β = 1 at θ = 0 freezes retained at the fair premium", () => {
    const p = { ...base, nPaths: 5_000, beta: 1 };
    const r = simulateRun(p);
    const s = summarize(r.retainedSamples);
    expect(s.variance).toBeLessThan(1e-12);
    const b2bMean = summarize(r.b2bSamples).mean;
    expect(s.mean).toBeCloseTo(b2bMean, 6);
  });

  it("CVaR mode differs from Sharpe mode by the Gaussian factor", () => {
    const GAUSS = 2.062713055949736;
    const shared = { ...base, nPaths: 4_000, seed: 777 };
    const rFair = simulateRun({ ...shared, beta: 0.4, premiumLoad: 0 });
    const rSharpe = simulateRun({
      ...shared, beta: 0.4, premiumLoad: 0.6, premiumMode: "sharpe",
    });
    const rCvar = simulateRun({
      ...shared, beta: 0.4, premiumLoad: 0.6, premiumMode: "cvar",
    });
    expect(rSharpe.premium).toBeLessThan(rFair.premium);
    const loadSharpe = rFair.premium - rSharpe.premium;
    const loadCvar = rFair.premium - rCvar.premium;
    expect(loadCvar / loadSharpe).toBeCloseTo(GAUSS, 8);
  });
});
