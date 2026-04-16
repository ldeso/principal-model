// Closed-form moments of the GBM path integral
//
//   I_T := ∫₀ᵀ S_t dt,    dS_t = μ S_t dt + σ S_t dW_t,  S_0 > 0.
//
// Source: Dufresne (2001). Mirrors the §1 formulas in research-note.md.
// The μ → 0 and (2μ + σ²) → 0 limits are handled via expm1(x)/x.

export interface GbmMoments {
  mean: number;
  variance: number;
  secondMoment: number;
}

// (e^x − 1) / x with the analytic limit 1 at x = 0. Numerically stable for
// small |x| because Math.expm1 preserves precision there.
export function expm1OverX(x: number): number {
  if (x === 0) return 1;
  if (Math.abs(x) < 1e-8) {
    // Taylor: 1 + x/2 + x²/6 + O(x³).
    return 1 + x / 2 + (x * x) / 6;
  }
  return Math.expm1(x) / x;
}

export function expectedIt(S0: number, mu: number, T: number): number {
  return S0 * T * expm1OverX(mu * T);
}

export function secondMomentIt(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): number {
  // σ = 0: process is deterministic, I_T = S_0 · T · (e^{μT} − 1)/(μT),
  // so E[I_T²] = E[I_T]². Handle directly to sidestep 0/0 when μ = 0.
  if (sigma === 0) {
    const m = expectedIt(S0, mu, T);
    return m * m;
  }

  const s2 = sigma * sigma;
  const a = mu;
  const b = 2 * mu + s2;
  const denom = mu + s2;

  // Inner bracket: (e^{bT} − 1)/b − (e^{aT} − 1)/a, stable via expm1OverX.
  const bracket = T * expm1OverX(b * T) - T * expm1OverX(a * T);

  if (Math.abs(denom) < 1e-12) {
    // μ ≈ −σ² with σ > 0. The bracket stays finite while the prefactor
    // 1/(μ + σ²) blows up, so take a single-variable L'Hopital in σ² at
    // fixed μ. Only (e^{bT} − 1)/b depends on σ² (through b = 2μ + σ²);
    // its derivative w.r.t. σ² is (T e^{bT} − (e^{bT} − 1)/b) / b. Evaluate
    // at b = μ:
    const aT = a * T;
    return 2 * S0 * S0 * ((T * Math.exp(aT)) / a - Math.expm1(aT) / (a * a));
  }

  return (2 * S0 * S0 * bracket) / denom;
}

export function varianceIt(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): number {
  const m1 = expectedIt(S0, mu, T);
  const m2 = secondMomentIt(S0, mu, sigma, T);
  // Floor at 0 to absorb catastrophic cancellation when σ is near-0.
  return Math.max(0, m2 - m1 * m1);
}

export function gbmMoments(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): GbmMoments {
  const mean = expectedIt(S0, mu, T);
  const secondMoment = secondMomentIt(S0, mu, sigma, T);
  const variance = Math.max(0, secondMoment - mean * mean);
  return { mean, variance, secondMoment };
}
