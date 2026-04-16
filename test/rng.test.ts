import { describe, expect, it } from "vitest";
import { mulberry32 } from "../src/rng.ts";

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
    expect(differ).toBeGreaterThan(90);
  });

  it("uniform output stays strictly in (0, 1)", () => {
    const rng = mulberry32(7);
    for (let i = 0; i < 10_000; i++) {
      const u = rng.uniform();
      expect(u).toBeGreaterThan(0);
      expect(u).toBeLessThan(1);
    }
  });

  it("normals have ~zero mean and ~unit variance", () => {
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
    // 4σ-ish tolerance: sd of sample mean ≈ 1/√n.
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(variance - 1)).toBeLessThan(0.02);
  });
});
