import { describe, expect, it } from "vitest";
import {
  conditionalVaR,
  shortfallVsSchedule,
  probLoss,
  quantile,
  summarise,
  summarize,
  valueAtRisk,
} from "../src/core/risk.js";

describe("quantile", () => {
  it("matches analytic quantiles at 0, 0.5, 1", () => {
    const xs = Float64Array.from([1, 2, 3, 4, 5]);
    expect(quantile(xs, 0)).toBe(1);
    expect(quantile(xs, 1)).toBe(5);
    expect(quantile(xs, 0.5)).toBe(3);
  });

  it("interpolates linearly between ranks", () => {
    const xs = [0, 10];
    expect(quantile(xs, 0.5)).toBeCloseTo(5);
    expect(quantile(xs, 0.25)).toBeCloseTo(2.5);
  });

  it("rejects invalid inputs", () => {
    expect(() => quantile([1, 2], -0.1)).toThrow();
    expect(() => quantile([1, 2], 1.1)).toThrow();
    expect(() => quantile([], 0.5)).toThrow();
  });
});

describe("valueAtRisk / conditionalVaR", () => {
  it("symmetric sample: VaR and CVaR match the tail cutoff", () => {
    const n = 10_000;
    const xs = new Float64Array(n);
    for (let i = 0; i < xs.length; i++) {
      xs[i] = -1 + (2 * i) / (xs.length - 1);
    }
    expect(valueAtRisk(xs, 0.95)).toBeCloseTo(0.9, 8);
    // k = ceil((1 − 0.95)·n) = 501 in IEEE-754.
    const k = Math.ceil((1 - 0.95) * n);
    const exactCvar = 1 - (k - 1) / (n - 1);
    expect(conditionalVaR(xs, 0.95)).toBeCloseTo(exactCvar, 10);
  });

  it("CVaR ≥ VaR always", () => {
    const xs = new Float64Array(1000);
    let state = 7;
    for (let i = 0; i < xs.length; i++) {
      state = (state * 1103515245 + 12345) >>> 0;
      xs[i] = (state / 2 ** 32) * 4 - 2;
    }
    for (const q of [0.9, 0.95, 0.99]) {
      expect(conditionalVaR(xs, q)).toBeGreaterThanOrEqual(valueAtRisk(xs, q));
    }
  });
});

describe("probLoss", () => {
  it("counts strictly negative outcomes", () => {
    expect(probLoss([-1, 0, 1, 2])).toBe(0.25);
    expect(probLoss([0, 0, 0])).toBe(0);
    expect(probLoss([-1, -2])).toBe(1);
  });
});

describe("shortfallVsSchedule", () => {
  it("zero when S is constant at S_0", () => {
    expect(shortfallVsSchedule([1, 1, 1, 1], 1000)).toBe(0);
  });

  it("zero when S only rises above S_0", () => {
    expect(shortfallVsSchedule([1, 1.1, 1.2, 1.3], 1000)).toBe(0);
  });

  it("zero at t = 0 and t = T (decay factor pins the ends)", () => {
    const S = [1, 0.5];
    expect(shortfallVsSchedule(S, 1000)).toBe(0);
  });

  it("scales with notional and (1 − t/T)(S_0 − S_t)", () => {
    // 5 nodes, S linear 1 → 0.5. At k = 2: decay 0.5, gap = 0.5·0.25·N.
    const S = [1.0, 0.875, 0.75, 0.625, 0.5];
    const N = 400;
    expect(shortfallVsSchedule(S, N)).toBeCloseTo(0.5 * 0.25 * N, 12);
  });
});

describe("summarize", () => {
  it("sample mean, n−1 variance, stderr", () => {
    const xs = [1, 2, 3, 4, 5];
    const s = summarize(xs);
    expect(s.mean).toBe(3);
    expect(s.variance).toBeCloseTo(2.5, 12);
    expect(s.sd).toBeCloseTo(Math.sqrt(2.5), 12);
    expect(s.stderr).toBeCloseTo(Math.sqrt(2.5 / 5), 12);
  });
});

describe("summarise", () => {
  const xs = new Float64Array(401);
  for (let i = 0; i < xs.length; i++) xs[i] = -2 + i * 0.01;

  it("agrees with summarize on mean / sd / stderr / ci95", () => {
    const s = summarise(xs);
    const base = summarize(xs);
    expect(s.mean).toBeCloseTo(base.mean, 12);
    expect(s.sd).toBeCloseTo(base.sd, 12);
    expect(s.stderr).toBeCloseTo(base.stderr, 12);
    expect(s.ci95).toBeCloseTo(base.ci95, 12);
  });

  it("routes VaR / CVaR through the primitives", () => {
    const s = summarise(xs);
    expect(s.var95).toBeCloseTo(valueAtRisk(xs, 0.95), 12);
    expect(s.var99).toBeCloseTo(valueAtRisk(xs, 0.99), 12);
    expect(s.cvar95).toBeCloseTo(conditionalVaR(xs, 0.95), 12);
    expect(s.cvar99).toBeCloseTo(conditionalVaR(xs, 0.99), 12);
  });

  it("matches probLoss; sharpe = mean/sd when sd > 0", () => {
    const s = summarise(xs);
    expect(s.probLoss).toBe(probLoss(xs));
    expect(s.sharpe).not.toBeNull();
    expect(s.sharpe as number).toBeCloseTo(s.mean / s.sd, 12);
  });

  it("sharpe = null for a constant sample", () => {
    const s = summarise([3, 3, 3, 3]);
    expect(s.sd).toBe(0);
    expect(s.sharpe).toBeNull();
    expect(s.probLoss).toBe(0);
  });

  it("does not throw on n = 1", () => {
    const s = summarise([1.5]);
    expect(s.mean).toBe(1.5);
    expect(s.sd).toBe(0);
    expect(s.sharpe).toBeNull();
  });
});
