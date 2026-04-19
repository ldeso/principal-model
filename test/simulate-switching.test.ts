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
// moderate nPaths for fast iteration, seed bumped off the defaults to keep
// test noise independent of the models.test.ts tape.
const base: SwitchingInputs = {
  S0: 1,
  mu: 0.05,
  sigma: 0.3,
  P: 1,
  lambda: 1_000,
  T: 1,
  Q: 1.1,
  fee: 0.05,
  alpha: 0,
  beta: 0,
  premiumLoad: 0,
  premiumMode: "sharpe",
  barrierRatio: Infinity,
  feePost: null,
  nPaths: 2_000,
  nSteps: 200,
  seed: 2026,
};

describe("simulateSwitching — §3e barrier-triggered mode switch", () => {
  it("barrierRatio = Infinity makes every path unswitched (τ = T)", () => {
    const r = simulateSwitching(base);
    for (let i = 0; i < r.switchedMask.length; i++) {
      expect(r.switchedMask[i]).toBe(0);
      expect(r.tauSamples[i]).toBe(base.T);
    }
    // With no switch, J_τ = 0 and the stochastic leg collapses onto b2b.
    for (let i = 0; i < r.stochLegSamples.length; i++) {
      expect(r.stochLegSamples[i]).toBeCloseTo(r.b2bSamples[i] as number, 10);
    }
  });

  it("barrierRatio ≤ 1 fires the switch at t = 0; stoch leg = fee × I_T", () => {
    const r = simulateSwitching({ ...base, barrierRatio: 1 });
    for (let i = 0; i < r.tauSamples.length; i++) {
      expect(r.tauSamples[i]).toBe(0);
      expect(r.switchedMask[i]).toBe(1);
      expect(r.ITauSamples[i]).toBe(0);
    }
    // stoch leg = f·P·λ·I_T = fee sample (since feePost locks to fee).
    for (let i = 0; i < r.stochLegSamples.length; i++) {
      expect(r.stochLegSamples[i]).toBeCloseTo(r.feeSamples[i] as number, 10);
    }
  });

  it("α = 1 makes Π_{3e} deterministic regardless of barrier", () => {
    const p = { ...base, alpha: 1, barrierRatio: 1.25 };
    const r = simulateSwitching(p);
    const expected = p.alpha * p.lambda * p.T * (p.Q - p.P * p.S0);
    // α=1 still enables the cession term on the (1-α)=0 stoch leg, so the
    // premium contribution is identically 0 and Π_{3e} collapses to α·matched.
    for (let i = 0; i < r.pnlSamples.length; i++) {
      expect(r.pnlSamples[i]).toBeCloseTo(expected, 6);
    }
  });

  it("hand-computes Π_sw = Q·λ·τ − P·λ·I_τ + f_post·P·λ·J_τ for a seeded sample", () => {
    // Use a small run so we can compare index-by-index and catch any
    // off-by-one in the trapezoid bucket assignment.
    const p = {
      ...base,
      alpha: 0,
      beta: 0,
      barrierRatio: 1.2,
      nPaths: 30,
      nSteps: 40,
      seed: 9999,
    };
    const r = simulateSwitching(p);
    const fPost = p.feePost ?? p.fee;
    for (let i = 0; i < r.stochLegSamples.length; i++) {
      const tau = r.tauSamples[i] as number;
      const ITau = r.ITauSamples[i] as number;
      const JTau = r.JTauSamples[i] as number;
      const expected =
        p.Q * p.lambda * tau -
        p.P * p.lambda * ITau +
        fPost * p.P * p.lambda * JTau;
      expect(r.stochLegSamples[i]).toBeCloseTo(expected, 9);
      // Sanity: I_τ + J_τ = I_T to machine precision.
      expect(ITau + JTau).toBeCloseTo(r.ITSamples[i] as number, 12);
    }
    // Same check with an independent feePost knob to exercise that code path.
    const r2 = simulateSwitching({ ...p, feePost: 0.15 });
    for (let i = 0; i < r2.stochLegSamples.length; i++) {
      const tau = r2.tauSamples[i] as number;
      const ITau = r2.ITauSamples[i] as number;
      const JTau = r2.JTauSamples[i] as number;
      const expected =
        p.Q * p.lambda * tau -
        p.P * p.lambda * ITau +
        0.15 * p.P * p.lambda * JTau;
      expect(r2.stochLegSamples[i]).toBeCloseTo(expected, 9);
    }
  });

  it("feePost = 0 reduces Π_sw to Q·λ·τ − P·λ·I_τ per path", () => {
    const p = { ...base, alpha: 0, beta: 0, barrierRatio: 1.2, feePost: 0 };
    const r = simulateSwitching(p);
    for (let i = 0; i < r.stochLegSamples.length; i++) {
      const tau = r.tauSamples[i] as number;
      const ITau = r.ITauSamples[i] as number;
      const expected = p.Q * p.lambda * tau - p.P * p.lambda * ITau;
      expect(r.stochLegSamples[i]).toBeCloseTo(expected, 9);
    }
  });

  it("feePost = f equals feePost = null path-by-path (lock-to-f)", () => {
    const shared = { ...base, alpha: 0.2, beta: 0.3, barrierRatio: 1.25 };
    const rNull = simulateSwitching({ ...shared, feePost: null });
    const rLocked = simulateSwitching({ ...shared, feePost: shared.fee });
    for (let i = 0; i < rNull.pnlSamples.length; i++) {
      expect(rNull.pnlSamples[i]).toBeCloseTo(rLocked.pnlSamples[i] as number, 10);
      expect(rNull.stochLegSamples[i]).toBeCloseTo(
        rLocked.stochLegSamples[i] as number,
        10,
      );
    }
    expect(rNull.feePostResolved).toBe(shared.fee);
    expect(rLocked.feePostResolved).toBe(shared.fee);
  });

  it("shares RNG with models.simulate under shared seed (I_T parity path-by-path)", () => {
    // Both runners consume the rng stream in the same order (one rng.normal()
    // per step plus the jump block). Under shared seed the I_T realisation
    // per path must match bit-for-bit — this guards path-reuse invariance.
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
      Q: p.Q, fee: p.f, alpha: p.alpha, beta: p.beta,
      premiumLoad: p.premiumLoad, premiumMode: p.premiumMode,
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
    // Path-reuse invariance: identical RNG consumption means identical S_t
    // realisations. Terminal prices agree to machine precision; I_T agrees
    // to 12 decimals (the two accumulators sum the same values in a slightly
    // different order — samplePath uses a single running acc, we use per-
    // step trapezoid pieces — so the last 2-3 ulps can differ).
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
    // Pick (h, μ, σ, T) triples where the barrier probability is non-degenerate;
    // nSteps = 1000 keeps the discrete-crossing bias below the stderr band.
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
      // Discrete-crossing bias is O(√Δt); at nSteps=1000 this is well under
      // the 4σ band for any of these barrier probabilities.
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
    // Strict monotone decrease. Finer discretisation can only add more
    // crossings (or the same count); under-estimation bias must shrink.
    expect(biases[1]).toBeLessThan(biases[0] as number);
    expect(biases[2]).toBeLessThan(biases[1] as number);
  });

  it("compensated Merton jumps strictly raise P[τ ≤ T] at equal σ", () => {
    // Jumps punch through the barrier in one step; under a fair barrier band
    // (not deep in-the-money), the empirical P[switch] must exceed the pure-
    // GBM anchor even though both preserve E[S_t].
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

  it("composes linearly with α and β: retained scaling matches the formula", () => {
    // Compare α=0.4, β=0.3, h=1.25 against the recomposition from the raw
    // stoch leg samples: Π_{3e} = α·N·(Q − P·S0) + (1−α)(1−β)·Π_sw + π_loaded.
    const p = { ...base, alpha: 0.4, beta: 0.3, barrierRatio: 1.25, premiumLoad: 0.6 };
    const r = simulateSwitching(p);
    const N = p.lambda * p.T;
    const matchedShift = p.alpha * N * (p.Q - p.P * p.S0);
    const scale = (1 - p.alpha) * (1 - p.beta);
    for (let i = 0; i < r.pnlSamples.length; i++) {
      const expected =
        matchedShift +
        scale * (r.stochLegSamples[i] as number) +
        r.premiumLoaded;
      expect(r.pnlSamples[i]).toBeCloseTo(expected, 10);
    }
  });

  it("fair-premium (θ = 0) keeps E[Π_{3e}] invariant in β", () => {
    const betas = [0, 0.25, 0.5, 0.75, 1];
    const shared = {
      ...base,
      alpha: 0.3, barrierRatio: 1.25, premiumLoad: 0,
      nPaths: 10_000, nSteps: 300, seed: 77,
    };
    const means: number[] = [];
    const cis: number[] = [];
    for (const beta of betas) {
      const r = simulateSwitching({ ...shared, beta });
      const s = summarize(r.pnlSamples);
      means.push(s.mean);
      cis.push(4 * s.stderr);
    }
    for (let i = 1; i < betas.length; i++) {
      const tol = Math.max(cis[0] as number, cis[i] as number);
      expect(Math.abs((means[i] as number) - (means[0] as number))).toBeLessThan(tol);
    }
  });

  it("CVaR decomposition has the right sign/order relative to the switched-mask partition", () => {
    // Conditional CVaRs must bracket the overall CVaR when the partitioning
    // is non-trivial (both buckets have mass). Exact partition-linearity does
    // not hold for CVaR; we only check that the overall CVaR sits between
    // the two conditional CVaRs (or equals one when the other is tiny).
    const p = {
      ...base,
      alpha: 0, beta: 0, barrierRatio: 1.25,
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
    // Small slack for sampling jitter on a ~5% tail.
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
    expect(standardNormalCdf(1.96)).toBeCloseTo(0.975, 3);
    expect(standardNormalCdf(-2)).toBeCloseTo(0.02275, 3);
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
