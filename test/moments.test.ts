import { describe, expect, it } from "vitest";
import {
  expectedIt,
  expm1OverX,
  gbmMoments,
  secondMomentIt,
  varianceIt,
} from "../src/moments.ts";

describe("expm1OverX", () => {
  it("returns 1 at x = 0", () => {
    expect(expm1OverX(0)).toBe(1);
  });

  it("matches Math.expm1(x)/x away from zero", () => {
    for (const x of [-2, -0.3, 0.01, 0.5, 1.7, 5]) {
      expect(expm1OverX(x)).toBeCloseTo(Math.expm1(x) / x, 12);
    }
  });

  it("is smooth across the series threshold (both sides ≈ 1)", () => {
    // |x| of order 1e-8 → analytic value within 1e-8 of 1. Test that the
    // branch split at |x| < 1e-8 doesn't introduce a discontinuity larger
    // than the expected linear-term contribution.
    const left = expm1OverX(1.1e-8);
    const right = expm1OverX(9e-9);
    expect(Math.abs(left - 1)).toBeLessThan(1e-7);
    expect(Math.abs(right - 1)).toBeLessThan(1e-7);
    // Difference is O(|x_left − x_right|) = O(2e-9).
    expect(Math.abs(left - right)).toBeLessThan(1e-8);
  });
});

describe("Dufresne moments", () => {
  it("E[I_T] → S_0 · T as μ → 0", () => {
    expect(expectedIt(1, 0, 3)).toBeCloseTo(3, 12);
    expect(expectedIt(2.5, 1e-12, 1.5)).toBeCloseTo(2.5 * 1.5, 10);
  });

  it("E[I_T] matches the Dufresne closed form for μ ≠ 0", () => {
    const S0 = 1.3;
    const mu = 0.08;
    const T = 2;
    const expected = (S0 * (Math.expm1(mu * T))) / mu;
    expect(expectedIt(S0, mu, T)).toBeCloseTo(expected, 12);
  });

  it("Var[I_T] = 0 when σ = 0 regardless of μ, S_0, T", () => {
    for (const mu of [-0.1, 0, 0.05, 0.2]) {
      for (const S0 of [0.5, 1, 3.2]) {
        for (const T of [0.5, 1, 2]) {
          expect(varianceIt(S0, mu, 0, T)).toBeLessThan(1e-10);
        }
      }
    }
  });

  it("Var[I_T] > 0 when σ > 0", () => {
    expect(varianceIt(1, 0.05, 0.5, 1)).toBeGreaterThan(0);
    expect(varianceIt(1, 0, 0.8, 2)).toBeGreaterThan(0);
  });

  it("matches the bundled gbmMoments helper", () => {
    const m = gbmMoments(1.4, 0.07, 0.55, 1.25);
    expect(m.mean).toBe(expectedIt(1.4, 0.07, 1.25));
    expect(m.secondMoment).toBe(secondMomentIt(1.4, 0.07, 0.55, 1.25));
    expect(m.variance).toBe(
      Math.max(0, secondMomentIt(1.4, 0.07, 0.55, 1.25) - m.mean * m.mean),
    );
  });
});
