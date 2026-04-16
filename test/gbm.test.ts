import { describe, expect, it } from "vitest";
import { samplePath, sampleTerminal } from "../src/gbm.ts";
import { expectedIt, varianceIt } from "../src/moments.ts";
import { mulberry32 } from "../src/rng.ts";
import { summarize } from "../src/risk.ts";

describe("samplePath", () => {
  it("degenerates to the deterministic curve when σ = 0", () => {
    const rng = mulberry32(1);
    const path = samplePath(rng, {
      S0: 1,
      mu: 0.1,
      sigma: 0,
      T: 1,
      nSteps: 100,
    });
    for (let k = 0; k <= 100; k++) {
      const tk = k / 100;
      expect(path.S[k]).toBeCloseTo(Math.exp(0.1 * tk), 10);
    }
    // Trapezoidal quadrature of e^{0.1 t} on [0,1] equals (e^{0.1} − 1)/0.1
    // up to O(Δ²) error.
    const exactIt = Math.expm1(0.1) / 0.1;
    expect(path.IT).toBeCloseTo(exactIt, 3);
  });

  it("terminal mean matches S_0 · e^{μT}", () => {
    const rng = mulberry32(2025);
    const n = 50_000;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = sampleTerminal(rng, {
        S0: 1,
        mu: 0.1,
        sigma: 0.4,
        T: 1,
        nSteps: 50,
      });
    }
    const stats = summarize(samples);
    const expected = Math.exp(0.1);
    expect(Math.abs(stats.mean - expected)).toBeLessThan(4 * stats.stderr);
  });

  it("MC E[I_T] and Var[I_T] match the Dufresne closed form within CI", () => {
    const S0 = 1;
    const mu = 0.05;
    const sigma = 0.5;
    const T = 1;
    const n = 100_000;

    const rng = mulberry32(11);
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = samplePath(rng, {
        S0,
        mu,
        sigma,
        T,
        nSteps: 200,
      }).IT;
    }
    const stats = summarize(samples);
    const meanCf = expectedIt(S0, mu, T);
    const varCf = varianceIt(S0, mu, sigma, T);

    // Mean: sample mean within 4 · stderr of the closed form.
    expect(Math.abs(stats.mean - meanCf)).toBeLessThan(4 * stats.stderr);

    // Variance: relative error under 5% at n = 100k. Sample variance stderr
    // is σ² · √(2/(n−1)) which, with Var[I_T] ~ 0.09 and n = 1e5, gives
    // ~0.4% — 5% leaves comfortable headroom for quadrature bias.
    expect(Math.abs(stats.variance - varCf) / varCf).toBeLessThan(0.05);
  });
});
