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

describe("simulateRun — Phase C custom-principal book", () => {
  it("kPre = 0 collapses principal onto b2b − cBasis", () => {
    // No pre-purchased inventory ⇒ tauFrac = 0 ⇒ tailInt = I_T ⇒
    // principal = Q·N − C_basis − P·λ·I_T = b2b − C_basis.
    const cBasis = 12;
    const r = simulateRun({ ...base, kPre: 0, cBasis });
    expect(r.tauFrac).toBe(0);
    expect(r.tokensLeftover).toBe(0);
    for (let i = 0; i < r.principalSamples.length; i++) {
      expect(r.principalSamples[i]).toBeCloseTo(
        (r.b2bSamples[i] as number) - cBasis,
        10,
      );
    }
  });

  it("kPre = λ·P·T (exact coverage) makes the principal book deterministic", () => {
    // Enough inventory to cover the whole horizon ⇒ tauFrac = 1 ⇒ tailInt = 0
    // ⇒ principal = Q·N − C_basis for every path.
    const p = { ...base, kPre: base.lambda * base.P * base.T, cBasis: 40 };
    const r = simulateRun(p);
    expect(r.tauFrac).toBe(1);
    expect(r.tokensLeftover).toBe(0);
    expect(r.tokensUsedInternal).toBe(p.kPre);
    const expected = p.Q * r.N - p.cBasis;
    for (let i = 0; i < r.principalSamples.length; i++) {
      expect(r.principalSamples[i]).toBe(expected);
    }
  });

  it("kPre > λ·P·T marks leftover tokens to spot at horizon", () => {
    // Double the needed inventory ⇒ half is retired, half survives to T and
    // is marked at S_T: principal = Q·N − C_basis + leftover·S_T.
    const covered = base.lambda * base.P * base.T;
    const p = { ...base, kPre: 2 * covered, cBasis: 77 };
    const r = simulateRun(p);
    expect(r.tauFrac).toBe(1);
    expect(r.tokensLeftover).toBe(covered);
    expect(r.tokensUsedInternal).toBe(covered);
    for (let i = 0; i < r.principalSamples.length; i++) {
      const residual = (r.principalSamples[i] as number) -
        r.tokensLeftover * (r.terminalS[i] as number);
      expect(residual).toBeCloseTo(p.Q * r.N - p.cBasis, 8);
    }
  });

  it("mean fee revenue matches the closed-form GBM anchor", () => {
    // Under pure GBM (λ_J = 0) E[I_T] = S_0·T·(e^{μT}−1)/(μT), so
    // E[R_fee] = f·P·λ·E[I_T]. Hold the MC mean to within 4 stderr.
    const p = { ...base, nPaths: 20_000, nSteps: 100 };
    const r = simulateRun(p);
    const eIT = expectedIt(p.S0, p.mu, p.T);
    const expectedFeeMean = p.fee * p.P * p.lambda * eIT;
    const s = summarize(r.feeSamples);
    expect(Math.abs(s.mean - expectedFeeMean)).toBeLessThan(4 * s.stderr);
  });

  it("shares the Merton kernel with samplePath (bit-for-bit under a common seed)", () => {
    // Same RNG stream, same path opts ⇒ simulateRun's I_T and S_T per path
    // must match an independent samplePath loop exactly.
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

  it("tauFrac = 1 when λ or P is zero (degenerate-demand guard)", () => {
    // If there's no demand, any pre-purchase covers the whole (zero) horizon
    // of retirements, so we never buy on the tail.
    const rZero = simulateRun({ ...base, lambda: 0, kPre: 0, cBasis: 0 });
    expect(rZero.tauFrac).toBe(1);
    expect(rZero.tokensUsedInternal).toBe(0);
    expect(rZero.tokensLeftover).toBe(0);
  });
});
