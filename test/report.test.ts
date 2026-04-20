import { describe, expect, it } from "vitest";
import {
  histogram,
  makeRow,
  makeSwitchingRow,
  subsample,
} from "../src/report.js";

describe("histogram", () => {
  it("degenerate sample (min == max) falls back to a single bin", () => {
    const xs = Float64Array.from([2, 2, 2, 2, 2]);
    const h = histogram(xs, 8);
    expect(h.edges).toEqual([2, 2]);
    expect(h.counts).toEqual([5]);
  });

  it("non-finite sample also folds to the single-bin path", () => {
    // All values NaN ⇒ lo stays Infinity, !isFinite(lo) triggers the guard.
    const xs = Float64Array.from([Number.NaN, Number.NaN]);
    const h = histogram(xs, 4);
    expect(h.counts).toEqual([xs.length]);
    expect(h.edges).toHaveLength(2);
  });

  it("bins are monotone and counts sum to n", () => {
    const xs = new Float64Array(1_000);
    for (let i = 0; i < xs.length; i++) xs[i] = i / (xs.length - 1);
    const nBins = 10;
    const h = histogram(xs, nBins);
    expect(h.edges).toHaveLength(nBins + 1);
    expect(h.counts).toHaveLength(nBins);
    for (let i = 1; i < h.edges.length; i++) {
      expect(h.edges[i]).toBeGreaterThan(h.edges[i - 1] as number);
    }
    const total = h.counts.reduce((a, b) => a + b, 0);
    expect(total).toBe(xs.length);
  });

  it("places the sample maximum in the last bin (b === nBins guard)", () => {
    // The raw bin index for v = hi is exactly nBins; without the floor-guard
    // on src/report.ts:130 the max would land in a phantom (nBins+1)-th bin.
    const xs = Float64Array.from([0, 0, 1]);
    const h = histogram(xs, 4);
    expect(h.counts).toHaveLength(4);
    const total = h.counts.reduce((a, b) => a + b, 0);
    expect(total).toBe(xs.length);
    expect(h.counts[h.counts.length - 1]).toBeGreaterThanOrEqual(1);
  });
});

describe("subsample", () => {
  it("returns at most the requested number of items", () => {
    const xs = new Float64Array(1_000);
    for (let i = 0; i < xs.length; i++) xs[i] = i;
    const out = subsample(xs, 50);
    expect(out.length).toBeLessThanOrEqual(50);
    expect(out[0]).toBe(0);
    // Step = floor(1000/50) = 20 ⇒ out[1] − out[0] = 20.
    expect((out[1] as number) - (out[0] as number)).toBe(20);
  });

  it("returns every element when requested size ≥ sample length", () => {
    const xs = Float64Array.from([3, 1, 4, 1, 5, 9]);
    const out = subsample(xs, 100);
    expect(out).toEqual([3, 1, 4, 1, 5, 9]);
  });

  it("handles empty input without throwing", () => {
    const out = subsample(new Float64Array(0), 10);
    expect(out).toEqual([]);
  });
});

describe("makeRow / makeSwitchingRow", () => {
  it("non-degenerate sample fills z-score and sharpe", () => {
    const xs = new Float64Array(1_000);
    for (let i = 0; i < xs.length; i++) xs[i] = i / (xs.length - 1);
    // Sample mean = 0.5, sd ≈ 0.289 ⇒ sharpe ≈ 1.73 regardless of closed anchor.
    const row = makeRow("demo", 0.5, 0.1, xs);
    expect(row.mcMean).toBeCloseTo(0.5, 12);
    // mean equals closed anchor ⇒ z-score is zero even with stderr > 0.
    expect(row.zScore).toBeCloseTo(0, 12);
    expect(row.sharpe).not.toBeNull();
    expect(row.sharpe as number).toBeCloseTo(row.mcMean / row.mcSd, 12);
  });

  it("constant sample ⇒ stderr = 0 ⇒ z-score routes through the guard", () => {
    // samples all equal ⇒ variance = 0 ⇒ stderr = 0; the guard on
    // src/report.ts:91 must return 0 instead of NaN/Infinity.
    const xs = Float64Array.from([1.5, 1.5, 1.5, 1.5]);
    const row = makeRow("flat", 10, 0, xs);
    expect(row.mcSd).toBe(0);
    expect(row.zScore).toBe(0);
    expect(row.sharpe).toBeNull();
  });

  it("makeSwitchingRow tags closed-form fields with NaN and zeros z-score", () => {
    const xs = Float64Array.from([0.1, 0.2, 0.3, 0.4]);
    const row = makeSwitchingRow(xs);
    expect(row.name).toBe("switching");
    expect(Number.isNaN(row.closedFormMean)).toBe(true);
    expect(Number.isNaN(row.closedFormSd)).toBe(true);
    expect(row.zScore).toBe(0);
    expect(row.mcMean).toBeCloseTo(0.25, 12);
  });

  it("makeSwitchingRow returns sharpe = null on a constant sample", () => {
    const xs = Float64Array.from([2, 2, 2, 2]);
    const row = makeSwitchingRow(xs);
    expect(row.mcSd).toBe(0);
    expect(row.sharpe).toBeNull();
  });
});
