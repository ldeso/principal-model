import { describe, expect, it } from "vitest";
import {
  closedForm,
  partialDeskClosedForm,
  simulate,
} from "../src/core/models.js";
import { covarITST } from "../src/core/moments.js";
import { defaultParams, withOverrides } from "../src/params.js";
import { conditionalVaR, summarize } from "../src/core/risk.js";

describe("operating books — closed form ↔ MC", () => {
  const p = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
    lambdaJ: 0,
    muJ: 0,
    sigmaJ: 0,
  });
  const cf = closedForm(p);
  const mc = simulate(p);

  it("E[R_fee] within 4·stderr", () => {
    const s = summarize(mc.fee);
    expect(Math.abs(s.mean - cf.fee.mean)).toBeLessThan(4 * s.stderr);
  });

  it("Var[R_fee] within 5%", () => {
    const s = summarize(mc.fee);
    expect(Math.abs(s.variance - cf.fee.variance) / cf.fee.variance).toBeLessThan(
      0.05,
    );
  });

  it("E[Π_b2b] within 4·stderr", () => {
    const s = summarize(mc.b2b);
    expect(Math.abs(s.mean - cf.b2b.mean)).toBeLessThan(4 * s.stderr);
  });

  it("Var[Π_b2b] / Var[R_fee] = (P/f)²", () => {
    const sB = summarize(mc.b2b);
    const sF = summarize(mc.fee);
    const ratio = sB.variance / sF.variance;
    const expectedRatio = (p.P / p.f) ** 2;
    expect(Math.abs(ratio - expectedRatio) / expectedRatio).toBeLessThan(0.01);
  });
});

describe("treasury — closed form ↔ MC", () => {
  const base = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
    lambdaJ: 0,
    muJ: 0,
    sigmaJ: 0,
  });

  it("α = 0 ⇒ treasury ≡ 0 path-wise", () => {
    const p = withOverrides(base, { alpha: 0, nPaths: 5_000, nSteps: 50 });
    const mc = simulate(p);
    for (let i = 0; i < mc.treasury.length; i++) {
      expect(mc.treasury[i]).toBe(0);
    }
  });

  it("α = 1 mean within 4·stderr", () => {
    const p = withOverrides(base, { alpha: 1 });
    const cf = closedForm(p);
    const mc = simulate(p);
    const s = summarize(mc.treasury);
    expect(Math.abs(s.mean - cf.treasury.mean)).toBeLessThan(4 * s.stderr);
  });

  it("α = 1 variance within 5%", () => {
    const p = withOverrides(base, { alpha: 1 });
    const cf = closedForm(p);
    const mc = simulate(p);
    const s = summarize(mc.treasury);
    expect(
      Math.abs(s.variance - cf.treasury.variance) / cf.treasury.variance,
    ).toBeLessThan(0.05);
  });

  it("α = 0.4 mean and variance match closed form", () => {
    const p = withOverrides(base, { alpha: 0.4 });
    const cf = closedForm(p);
    const mc = simulate(p);
    const s = summarize(mc.treasury);
    expect(Math.abs(s.mean - cf.treasury.mean)).toBeLessThan(4 * s.stderr);
    expect(
      Math.abs(s.variance - cf.treasury.variance) / cf.treasury.variance,
    ).toBeLessThan(0.05);
  });

  it("mean invariant under compensated Merton", () => {
    const gbm = simulate(withOverrides(base, { alpha: 0.5 }));
    const merton = simulate(
      withOverrides(base, {
        alpha: 0.5,
        lambdaJ: 3,
        muJ: -0.1,
        sigmaJ: 0.15,
        seed: 2026,
      }),
    );
    const sG = summarize(gbm.treasury);
    const sM = summarize(merton.treasury);
    const tol = 4 * Math.max(sG.stderr, sM.stderr);
    expect(Math.abs(sG.mean - sM.mean)).toBeLessThan(tol);
  });
});

describe("desk compositions", () => {
  const base = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
    lambdaJ: 0,
    muJ: 0,
    sigmaJ: 0,
  });

  it("α = 1 ⇒ b2b + treasury = N·(Q − P·S_0) path-wise", () => {
    const p = withOverrides(base, { alpha: 1, nPaths: 10_000, nSteps: 50 });
    const mc = simulate(p);
    const expected = p.lambda * p.T * (p.Q - p.P * p.S0);
    let maxDeviation = 0;
    for (let i = 0; i < mc.b2b.length; i++) {
      const desk = (mc.b2b[i] as number) + (mc.treasury[i] as number);
      const d = Math.abs(desk - expected);
      if (d > maxDeviation) maxDeviation = d;
    }
    expect(maxDeviation).toBeLessThan(1e-8);
  });

  it("partial-desk mean within 4·stderr", () => {
    const p = withOverrides(base, { alpha: 0.4 });
    const cf = partialDeskClosedForm(p);
    const mc = simulate(p);
    const desk = new Float64Array(mc.b2b.length);
    for (let i = 0; i < desk.length; i++) {
      desk[i] = (mc.b2b[i] as number) + (mc.treasury[i] as number);
    }
    const s = summarize(desk);
    expect(Math.abs(s.mean - cf.mean)).toBeLessThan(4 * s.stderr);
  });

  it("partial-desk variance within 5%", () => {
    const p = withOverrides(base, { alpha: 0.4 });
    const cf = partialDeskClosedForm(p);
    const mc = simulate(p);
    const desk = new Float64Array(mc.b2b.length);
    for (let i = 0; i < desk.length; i++) {
      desk[i] = (mc.b2b[i] as number) + (mc.treasury[i] as number);
    }
    const s = summarize(desk);
    expect(Math.abs(s.variance - cf.variance) / cf.variance).toBeLessThan(0.05);
  });

  it("Cov[I_T, S_T] within 5%", () => {
    const p = withOverrides(base, { nPaths: 100_000, nSteps: 250 });
    const mc = simulate(p);
    let meanIT = 0;
    let meanST = 0;
    for (let i = 0; i < mc.IT.length; i++) {
      meanIT += mc.IT[i] as number;
      meanST += mc.terminalS[i] as number;
    }
    meanIT /= mc.IT.length;
    meanST /= mc.terminalS.length;
    let cov = 0;
    for (let i = 0; i < mc.IT.length; i++) {
      cov += ((mc.IT[i] as number) - meanIT) * ((mc.terminalS[i] as number) - meanST);
    }
    cov /= mc.IT.length - 1;
    const cfCov = covarITST(p.S0, p.mu, p.sigma, p.T);
    expect(Math.abs(cov - cfCov) / Math.abs(cfCov)).toBeLessThan(0.05);
  });
});

describe("break-even quote Q*", () => {
  it("equalises E[R_fee] and E[Π_b2b]", () => {
    const p = defaultParams;
    const cf = closedForm(p);
    const N = p.lambda * p.T;
    const b2bAtQstar = cf.QStar * N - p.P * p.lambda * cf.IT.mean;
    expect(b2bAtQstar).toBeCloseTo(cf.fee.mean, 9);
  });

  it("→ (1 + f)·P·S_0 as μ → 0", () => {
    const p = withOverrides(defaultParams, { mu: 0 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeCloseTo((1 + p.f) * p.P * p.S0, 12);
  });

  it("> (1 + f)·P·S_0 for μ > 0", () => {
    const p = withOverrides(defaultParams, { mu: 0.15 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeGreaterThan((1 + p.f) * p.P * p.S0);
  });

  it("< (1 + f)·P·S_0 for μ < 0", () => {
    const p = withOverrides(defaultParams, { mu: -0.1 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeLessThan((1 + p.f) * p.P * p.S0);
  });

  it("is β-invariant", () => {
    const p = defaultParams;
    const q0 = closedForm(withOverrides(p, { beta: 0 })).QStar;
    for (const beta of [0.25, 0.5, 0.75, 1]) {
      const q = closedForm(withOverrides(p, { beta })).QStar;
      expect(q).toBe(q0);
    }
  });
});

describe("matched desk identity", () => {
  it("closed-form matched desk = N·(Q − P·S_0), variance 0", () => {
    const p = withOverrides(defaultParams, { alpha: 1 });
    const cf = closedForm(p);
    const expected = p.lambda * p.T * (p.Q - p.P * p.S0);
    const partial = partialDeskClosedForm(p);
    expect(partial.mean).toBeCloseTo(expected, 9);
    expect(partial.variance).toBeLessThan(1e-12);
    expect(cf.b2b.mean + cf.treasury.mean).toBeCloseTo(expected, 9);
  });
});

describe("syndicated-on-b2b operating book", () => {
  const base = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
    lambdaJ: 0,
    muJ: 0,
    sigmaJ: 0,
  });

  it("β = 0 ⇒ retained ≡ b2b, premium 0", () => {
    const p = withOverrides(base, { beta: 0, premiumLoad: 0 });
    const mc = simulate(p);
    const cf = closedForm(p);
    for (let i = 0; i < mc.retained.length; i++) {
      expect(mc.retained[i]).toBeCloseTo(mc.b2b[i] as number, 12);
    }
    expect(cf.retained.mean).toBeCloseTo(cf.b2b.mean, 12);
    expect(cf.retained.variance).toBeCloseTo(cf.b2b.variance, 12);
    expect(cf.premium.fair).toBe(0);
    expect(cf.premium.loaded).toBe(0);
  });

  it("β = 1 at θ = 0 collapses to the fair premium, variance 0", () => {
    const p = withOverrides(base, { beta: 1, premiumLoad: 0, nPaths: 10_000 });
    const mc = simulate(p);
    const cf = closedForm(p);
    const s = summarize(mc.retained);
    expect(s.variance).toBeLessThan(1e-12);
    expect(s.mean).toBeCloseTo(cf.premium.loaded, 10);
    expect(cf.premium.fair).toBeCloseTo(cf.b2b.mean, 10);
  });

  it("mean and variance match closed form for arbitrary (β, θ)", () => {
    for (const mode of ["sharpe", "cvar"] as const) {
      const p = withOverrides(base, {
        beta: 0.3, premiumLoad: 0.5, premiumMode: mode,
      });
      const mc = simulate(p);
      const cf = closedForm(p);
      const s = summarize(mc.retained);
      expect(Math.abs(s.mean - cf.retained.mean)).toBeLessThan(4 * s.stderr);
      expect(
        Math.abs(s.variance - cf.retained.variance) / cf.retained.variance,
      ).toBeLessThan(0.05);
    }
  });

  it("variance collapses as (1 − β)²·Var[Π_b2b]", () => {
    const p = withOverrides(base, { beta: 0.6 });
    const cf = closedForm(p);
    const expected = (1 - p.beta) ** 2 * cf.b2b.variance;
    const mc = simulate(p);
    const s = summarize(mc.retained);
    expect(Math.abs(s.variance - expected) / expected).toBeLessThan(0.05);
  });

  it("fair-premium mean is β-invariant at θ = 0", () => {
    const betas = [0, 0.25, 0.5, 0.75, 1];
    const means: number[] = [];
    const cis: number[] = [];
    for (const beta of betas) {
      const p = withOverrides(base, { beta, premiumLoad: 0 });
      const mc = simulate(p);
      const s = summarize(mc.retained);
      means.push(s.mean);
      cis.push(4 * s.stderr);
    }
    for (let i = 1; i < betas.length; i++) {
      const tol = Math.max(cis[0] as number, cis[i] as number);
      expect(Math.abs((means[i] as number) - (means[0] as number))).toBeLessThan(tol);
    }
  });

  it("CVaR₉₅ non-increasing in β at θ = 0", () => {
    const betas = [0, 0.25, 0.5, 0.75, 1];
    const cvars: number[] = [];
    for (const beta of betas) {
      const p = withOverrides(base, { beta, premiumLoad: 0 });
      const mc = simulate(p);
      cvars.push(conditionalVaR(mc.retained, 0.95));
    }
    for (let i = 1; i < betas.length; i++) {
      const slack = 0.01 * Math.abs(cvars[0] as number);
      expect(cvars[i]).toBeLessThan((cvars[i - 1] as number) + slack);
    }
  });

  it("CVaR-mode premium = Gaussian factor × Sharpe-mode load", () => {
    const GAUSS = 2.062713055949736;
    const pSharpe = withOverrides(base, {
      beta: 0.4, premiumLoad: 0.7, premiumMode: "sharpe",
    });
    const pCvar = withOverrides(pSharpe, { premiumMode: "cvar" });
    const cfS = closedForm(pSharpe);
    const cfC = closedForm(pCvar);
    expect(cfS.premium.fair).toBeCloseTo(cfC.premium.fair, 12);
    const loadS = cfS.premium.fair - cfS.premium.loaded;
    const loadC = cfC.premium.fair - cfC.premium.loaded;
    expect(loadC / loadS).toBeCloseTo(GAUSS, 10);
  });
});
