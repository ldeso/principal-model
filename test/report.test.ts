import { describe, expect, it } from "vitest";
import {
  histogram,
  makeRow,
  makeSwitchingRow,
  subsample,
} from "../src/report.js";

describe("histogram", () => {
  it("degenerate sample (min == max) folds to one bin", () => {
    const xs = Float64Array.from([2, 2, 2, 2, 2]);
    const h = histogram(xs, 8);
    expect(h.edges).toEqual([2, 2]);
    expect(h.counts).toEqual([5]);
  });

  it("non-finite sample folds to one bin", () => {
    const xs = Float64Array.from([Number.NaN, Number.NaN]);
    const h = histogram(xs, 4);
    expect(h.counts).toEqual([xs.length]);
    expect(h.edges).toHaveLength(2);
  });

  it("bins monotone; counts sum to n", () => {
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

  it("places the sample maximum in the last bin", () => {
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
    expect((out[1] as number) - (out[0] as number)).toBe(20);
  });

  it("returns everything when requested size ≥ sample length", () => {
    const xs = Float64Array.from([3, 1, 4, 1, 5, 9]);
    const out = subsample(xs, 100);
    expect(out).toEqual([3, 1, 4, 1, 5, 9]);
  });

  it("handles empty input", () => {
    const out = subsample(new Float64Array(0), 10);
    expect(out).toEqual([]);
  });
});

describe("makeRow / makeSwitchingRow", () => {
  it("non-degenerate sample fills z-score and sharpe", () => {
    const xs = new Float64Array(1_000);
    for (let i = 0; i < xs.length; i++) xs[i] = i / (xs.length - 1);
    const row = makeRow("demo", 0.5, 0.1, xs);
    expect(row.mcMean).toBeCloseTo(0.5, 12);
    expect(row.zScore).toBeCloseTo(0, 12);
    expect(row.sharpe).not.toBeNull();
    expect(row.sharpe as number).toBeCloseTo(row.mcMean / row.mcSd, 12);
  });

  it("constant sample ⇒ stderr = 0 ⇒ zScore = 0", () => {
    const xs = Float64Array.from([1.5, 1.5, 1.5, 1.5]);
    const row = makeRow("flat", 10, 0, xs);
    expect(row.mcSd).toBe(0);
    expect(row.zScore).toBe(0);
    expect(row.sharpe).toBeNull();
  });

  it("makeSwitchingRow: closed-form fields NaN, zScore 0", () => {
    const xs = Float64Array.from([0.1, 0.2, 0.3, 0.4]);
    const row = makeSwitchingRow(xs);
    expect(row.name).toBe("switching");
    expect(Number.isNaN(row.closedFormMean)).toBe(true);
    expect(Number.isNaN(row.closedFormSd)).toBe(true);
    expect(row.zScore).toBe(0);
    expect(row.mcMean).toBeCloseTo(0.25, 12);
  });

  it("makeSwitchingRow: sharpe null on constant sample", () => {
    const xs = Float64Array.from([2, 2, 2, 2]);
    const row = makeSwitchingRow(xs);
    expect(row.mcSd).toBe(0);
    expect(row.sharpe).toBeNull();
  });
});
