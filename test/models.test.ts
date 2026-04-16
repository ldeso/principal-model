import { describe, expect, it } from "vitest";
import { closedForm, simulate } from "../src/models.ts";
import { defaultParams, withOverrides } from "../src/params.ts";
import { summarize } from "../src/risk.ts";

describe("closed-form ↔ Monte Carlo cross-check", () => {
  const p = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
  });
  const cf = closedForm(p);
  const mc = simulate(p);

  it("E[R_fee] agrees within 4 stderr", () => {
    const s = summarize(mc.fee);
    expect(Math.abs(s.mean - cf.fee.mean)).toBeLessThan(4 * s.stderr);
  });

  it("Var[R_fee] matches closed form within 5%", () => {
    const s = summarize(mc.fee);
    expect(Math.abs(s.variance - cf.fee.variance) / cf.fee.variance).toBeLessThan(
      0.05,
    );
  });

  it("E[Π_b2b] agrees within 4 stderr", () => {
    const s = summarize(mc.b2b);
    expect(Math.abs(s.mean - cf.b2b.mean)).toBeLessThan(4 * s.stderr);
  });

  it("Var[Π_b2b] = (P · λ)² · Var[I_T], ~ (P/f)² · Var[R_fee]", () => {
    // Research note §3b observation: the two books share the I_T kernel.
    const sB = summarize(mc.b2b);
    const sF = summarize(mc.fee);
    const ratio = sB.variance / sF.variance;
    const expectedRatio = (p.P / p.f) ** 2;
    expect(Math.abs(ratio - expectedRatio) / expectedRatio).toBeLessThan(0.01);
  });

  it("E[Π_α] interpolates linearly in α", () => {
    const sP = summarize(mc.partial);
    const expected =
      (1 - p.alpha) * cf.b2b.mean + p.alpha * cf.matched.mean;
    expect(Math.abs(sP.mean - expected)).toBeLessThan(4 * sP.stderr);
  });

  it("Var[Π_α] = (1 − α)² · Var[Π_b2b]", () => {
    const sP = summarize(mc.partial);
    const expected = (1 - p.alpha) ** 2 * cf.b2b.variance;
    expect(Math.abs(sP.variance - expected) / expected).toBeLessThan(0.05);
  });

  it("α = 1 collapses Π_α to the deterministic matched P&L", () => {
    const p1 = withOverrides(p, { alpha: 1, nPaths: 10_000, nSteps: 50 });
    const mc1 = simulate(p1);
    let maxDeviation = 0;
    for (let i = 0; i < mc1.partial.length; i++) {
      const v = mc1.partial[i] as number;
      const d = Math.abs(v - mc1.matched);
      if (d > maxDeviation) maxDeviation = d;
    }
    expect(maxDeviation).toBeLessThan(1e-8);
  });

  it("α = 0 makes Π_α coincide with Π_b2b path-by-path", () => {
    const p0 = withOverrides(p, { alpha: 0, nPaths: 5_000, nSteps: 50 });
    const mc0 = simulate(p0);
    for (let i = 0; i < mc0.partial.length; i++) {
      expect(mc0.partial[i]).toBeCloseTo(mc0.b2b[i] as number, 10);
    }
  });
});

describe("break-even quote Q*", () => {
  it("equalises E[R_fee] and E[Π_b2b] at Q = Q*", () => {
    const p = defaultParams;
    const cf = closedForm(p);
    const N = p.lambda * p.T;
    const b2bAtQstar = cf.QStar * N - p.P * p.lambda * cf.IT.mean;
    expect(b2bAtQstar).toBeCloseTo(cf.fee.mean, 9);
  });

  it("reduces to (1 + f) · P · S_0 as μ → 0", () => {
    const p = withOverrides(defaultParams, { mu: 0 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeCloseTo((1 + p.f) * p.P * p.S0, 12);
  });

  it("exceeds (1 + f) · P · S_0 for μ > 0", () => {
    const p = withOverrides(defaultParams, { mu: 0.15 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeGreaterThan((1 + p.f) * p.P * p.S0);
  });

  it("falls below (1 + f) · P · S_0 for μ < 0", () => {
    const p = withOverrides(defaultParams, { mu: -0.1 });
    const cf = closedForm(p);
    expect(cf.QStar).toBeLessThan((1 + p.f) * p.P * p.S0);
  });
});

describe("3a deterministic P&L", () => {
  it("reports zero variance in closed form", () => {
    const cf = closedForm(defaultParams);
    expect(cf.matched.variance).toBe(0);
    expect(cf.matched.sd).toBe(0);
  });
});
