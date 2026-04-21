// Mulberry32 PRNG, Box-Muller normals, Knuth product-of-uniforms Poisson.

export interface Rng {
  uniform(): number;
  normal(): number;
  /** Non-negative integer ~ Poisson(λ). Requires λ ≥ 0. */
  poisson(lambda: number): number;
}

export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  let cached: number | null = null;

  const uniform = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return u === 0 ? 1 / 4294967296 : u;
  };

  const normal = (): number => {
    if (cached !== null) {
      const z = cached;
      cached = null;
      return z;
    }
    const u1 = uniform();
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };

  const poisson = (lambda: number): number => {
    if (!(lambda > 0)) return 0;
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    for (;;) {
      k++;
      p *= uniform();
      if (p <= L) return k - 1;
    }
  };

  return { uniform, normal, poisson };
}
