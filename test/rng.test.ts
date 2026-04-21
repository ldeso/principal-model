import { describe, expect, it } from "vitest";
import { mulberry32 } from "../src/core/rng.js";
import { summarize } from "../src/core/risk.js";

describe("mulberry32", () => {
  it("is deterministic for a fixed seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 1_000; i++) {
      expect(a.uniform()).toBe(b.uniform());
    }
  });

  it("produces distinct streams for different seeds", () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    let differ = 0;
    for (let i = 0; i < 100; i++) {
      if (a.uniform() !== b.uniform()) differ++;
    }
    expect(differ).toBe(100);
  });

  it("uniform output stays in (0, 1)", () => {
    const rng = mulberry32(7);
    for (let i = 0; i < 10_000; i++) {
      const u = rng.uniform();
      expect(u).toBeGreaterThan(0);
      expect(u).toBeLessThan(1);
    }
  });

  it("normals have ≈ zero mean and unit variance", () => {
    const rng = mulberry32(2024);
    const n = 100_000;
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < n; i++) {
      const z = rng.normal();
      s += z;
      s2 += z * z;
    }
    const mean = s / n;
    const variance = s2 / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.015);
    expect(Math.abs(variance - 1)).toBeLessThan(0.015);
  });
});

describe("mulberry32.poisson", () => {
  it("returns 0 for λ ≤ 0 without consuming uniforms", () => {
    const rng = mulberry32(11);
    expect(rng.poisson(0)).toBe(0);
    expect(rng.poisson(-1)).toBe(0);
    expect(rng.poisson(Number.NaN)).toBe(0);
    const fresh = mulberry32(11);
    expect(rng.uniform()).toBe(fresh.uniform());
  });

  it("mean and variance match λ for small λ", () => {
    const rng = mulberry32(2024);
    const n = 200_000;
    const lambda = 0.05;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) samples[i] = rng.poisson(lambda);
    const s = summarize(samples);
    expect(Math.abs(s.mean - lambda)).toBeLessThan(4 * Math.sqrt(lambda / n));
    expect(Math.abs(s.variance - lambda) / lambda).toBeLessThan(0.05);
  });

  it("mean and variance match λ for moderate λ", () => {
    const rng = mulberry32(987);
    const n = 50_000;
    const lambda = 3;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) samples[i] = rng.poisson(lambda);
    const s = summarize(samples);
    expect(Math.abs(s.mean - lambda)).toBeLessThan(4 * Math.sqrt(lambda / n));
    expect(Math.abs(s.variance - lambda) / lambda).toBeLessThan(0.05);
  });

  it("outputs are non-negative integers", () => {
    const rng = mulberry32(3);
    for (let i = 0; i < 1_000; i++) {
      const k = rng.poisson(0.5);
      expect(Number.isInteger(k)).toBe(true);
      expect(k).toBeGreaterThanOrEqual(0);
    }
  });
});
