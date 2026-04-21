import { describe, expect, it } from "vitest";
import { samplePath } from "../src/core/gbm.js";
import { expectedIt, varianceIt } from "../src/core/moments.js";
import { mulberry32 } from "../src/core/rng.js";
import { summarize } from "../src/core/risk.js";

describe("samplePath with Merton jumps", () => {
  it("λ_J = 0 reproduces pure GBM bit-for-bit", () => {
    const base = { S0: 1.3, mu: 0.07, sigma: 0.35, T: 1.25, nSteps: 64 };
    const a = mulberry32(77);
    const b = mulberry32(77);
    const pathGbm = samplePath(a, base);
    const pathJumpZero = samplePath(b, {
      ...base,
      lambdaJ: 0,
      muJ: -0.3,
      sigmaJ: 0.2,
    });
    for (let k = 0; k <= base.nSteps; k++) {
      expect(pathJumpZero.S[k]).toBe(pathGbm.S[k]);
    }
    expect(pathJumpZero.IT).toBe(pathGbm.IT);
  });

  it("E[S_T] matches S_0·e^{μT} under compensated jumps", () => {
    const rng = mulberry32(2026);
    const S0 = 1;
    const mu = 0.1;
    const T = 1;
    const nSteps = 50;
    const n = 80_000;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const path = samplePath(rng, {
        S0,
        mu,
        sigma: 0.3,
        T,
        nSteps,
        lambdaJ: 4,
        muJ: -0.15,
        sigmaJ: 0.25,
      });
      samples[i] = path.S[nSteps] as number;
    }
    const stats = summarize(samples);
    const expected = S0 * Math.exp(mu * T);
    expect(Math.abs(stats.mean - expected)).toBeLessThan(4 * stats.stderr);
  });

  it("E[I_T] matches GBM closed form under compensated jumps", () => {
    const rng = mulberry32(31);
    const S0 = 1;
    const mu = 0.05;
    const T = 1;
    const nSteps = 100;
    const n = 80_000;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = samplePath(rng, {
        S0,
        mu,
        sigma: 0.4,
        T,
        nSteps,
        lambdaJ: 3,
        muJ: -0.2,
        sigmaJ: 0.2,
      }).IT;
    }
    const stats = summarize(samples);
    const meanCf = expectedIt(S0, mu, T);
    expect(Math.abs(stats.mean - meanCf)).toBeLessThan(4 * stats.stderr);
  });

  it("jumps inflate Var[I_T] above the GBM anchor", () => {
    const S0 = 1;
    const mu = 0.05;
    const sigma = 0.3;
    const T = 1;
    const nSteps = 100;
    const n = 60_000;

    const rng = mulberry32(911);
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = samplePath(rng, {
        S0,
        mu,
        sigma,
        T,
        nSteps,
        lambdaJ: 3,
        muJ: -0.3,
        sigmaJ: 0.2,
      }).IT;
    }
    const stats = summarize(samples);
    const varGbm = varianceIt(S0, mu, sigma, T);
    expect(stats.variance).toBeGreaterThan(1.5 * varGbm);
  });
});
