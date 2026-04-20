import { describe, expect, it } from "vitest";
import { closedForm, simulate } from "../src/core/models.js";
import { defaultParams, withOverrides } from "../src/params.js";
import { conditionalVaR, summarize } from "../src/core/risk.js";

describe("closed-form ↔ Monte Carlo cross-check", () => {
  // gbmMoments is pure-GBM; zero the Merton slice so closed-form variance
  // matches MC variance. The jump-aware paths are exercised in jump-gbm.test.
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
    // Research note back-to-back observation: the two books share the I_T kernel.
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

describe("Syndicated variant — quota-share syndication (β)", () => {
  // Pure GBM baseline so the closed-form variance is exact under MC.
  const base = withOverrides(defaultParams, {
    nPaths: 50_000,
    nSteps: 200,
    seed: 2026,
    lambdaJ: 0,
    muJ: 0,
    sigmaJ: 0,
  });

  it("β = 0 reduces Π_ret to Π_α path-by-path", () => {
    // With no cession and zero load the retained book must coincide with the
    // existing partial book bit-for-bit; closed-form moments agree.
    const p = withOverrides(base, { beta: 0, premiumLoad: 0 });
    const mc = simulate(p);
    const cf = closedForm(p);
    for (let i = 0; i < mc.retained.length; i++) {
      expect(mc.retained[i]).toBeCloseTo(mc.partial[i] as number, 12);
    }
    expect(cf.retained.mean).toBeCloseTo(cf.partial.mean, 12);
    expect(cf.retained.variance).toBeCloseTo(cf.partial.variance, 12);
    expect(cf.premium.fair).toBe(0);
    expect(cf.premium.loaded).toBe(0);
  });

  it("β = 1 at θ = 0 collapses to α·matched + fair premium, zero variance", () => {
    // Ceding the whole stochastic leg at the fair price leaves only the
    // deterministic α-matched cash-flow plus the up-front premium.
    const p = withOverrides(base, { alpha: 0.3, beta: 1, premiumLoad: 0, nPaths: 10_000 });
    const mc = simulate(p);
    const cf = closedForm(p);
    const s = summarize(mc.retained);
    expect(s.variance).toBeLessThan(1e-12);
    const expected =
      p.alpha * p.lambda * p.T * (p.Q - p.P * p.S0) + cf.premium.loaded;
    expect(s.mean).toBeCloseTo(expected, 10);
  });

  it("mean and variance agree with closed form for arbitrary (α, β, θ)", () => {
    for (const mode of ["sharpe", "cvar"] as const) {
      const p = withOverrides(base, {
        alpha: 0.4, beta: 0.3, premiumLoad: 0.5, premiumMode: mode,
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

  it("variance collapses as (1−α)²(1−β)²·Var[Π_b2b]", () => {
    const p = withOverrides(base, { alpha: 0.25, beta: 0.6 });
    const cf = closedForm(p);
    const expected = ((1 - p.alpha) * (1 - p.beta)) ** 2 * cf.b2b.variance;
    const mc = simulate(p);
    const s = summarize(mc.retained);
    expect(Math.abs(s.variance - expected) / expected).toBeLessThan(0.05);
  });

  it("fair-premium mean invariance in β (θ = 0)", () => {
    // At θ = 0 the actuarially fair premium exactly replaces ceded expected
    // P&L, so E[Π_ret] is independent of β — the "no free lunch" check.
    const betas = [0, 0.25, 0.5, 0.75, 1];
    const means: number[] = [];
    const cis: number[] = [];
    for (const beta of betas) {
      const p = withOverrides(base, { alpha: 0.3, beta, premiumLoad: 0 });
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

  it("CVaR₉₅ is non-increasing in β at θ = 0", () => {
    // Fair quota-share shrinks the loss tail proportionally to (1−β), so
    // CVaR₉₅ of the retained book should be monotone down to α·matched.
    const betas = [0, 0.25, 0.5, 0.75, 1];
    const cvars: number[] = [];
    for (const beta of betas) {
      const p = withOverrides(base, { alpha: 0.2, beta, premiumLoad: 0 });
      const mc = simulate(p);
      cvars.push(conditionalVaR(mc.retained, 0.95));
    }
    for (let i = 1; i < betas.length; i++) {
      // Small MC jitter allowed — require strict monotonicity within 1% of
      // the β = 0 baseline magnitude.
      const slack = 0.01 * Math.abs(cvars[0] as number);
      expect(cvars[i]).toBeLessThan((cvars[i - 1] as number) + slack);
    }
  });

  it("Q* is invariant in β", () => {
    // Q* is defined by E[R_fee] = E[Π_b2b] and touches neither α nor β.
    const q0 = closedForm(withOverrides(base, { beta: 0 })).QStar;
    for (const beta of [0.25, 0.5, 0.75, 1]) {
      const q = closedForm(withOverrides(base, { beta })).QStar;
      expect(q).toBe(q0);
    }
  });

  it("CVaR-mode premium scales Sharpe-mode load by the Gaussian CVaR factor", () => {
    // Algebraic identity on the closed-form premium scalars: the two modes
    // differ only in which risk measure multiplies θ.
    const GAUSS = 2.062713055949736;
    const pSharpe = withOverrides(base, {
      alpha: 0.2, beta: 0.4, premiumLoad: 0.7, premiumMode: "sharpe",
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
