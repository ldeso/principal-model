// Closed-form moments of the GBM path integral I_T := ∫₀ᵀ S_t dt.
// Source: Dufresne (2001); μ → 0 and (2μ + σ²) → 0 limits via expm1(x)/x.

export interface GbmMoments {
  mean: number;
  variance: number;
}

// (e^x − 1) / x, with the analytic limit 1 at x = 0. expm1 keeps precision
// for small |x|; the series branch covers values where expm1(x)/x loses ulps.
export function expm1OverX(x: number): number {
  if (x === 0) return 1;
  if (Math.abs(x) < 1e-8) return 1 + x / 2 + (x * x) / 6;
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
  // σ = 0: deterministic, so E[I_T²] = E[I_T]². Handle directly to avoid 0/0.
  if (sigma === 0) {
    const m = expectedIt(S0, mu, T);
    return m * m;
  }

  const s2 = sigma * sigma;
  const a = mu;
  const b = 2 * mu + s2;
  const denom = mu + s2;

  const bracket = T * expm1OverX(b * T) - T * expm1OverX(a * T);

  if (Math.abs(denom) < 1e-12) {
    // μ ≈ −σ² with σ > 0: 1/(μ+σ²) diverges while the bracket stays finite.
    // L'Hôpital in σ² at fixed μ, evaluated at b = μ.
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
  // Floor at 0 to absorb catastrophic cancellation near σ = 0.
  return Math.max(0, m2 - m1 * m1);
}

export function gbmMoments(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): GbmMoments {
  return {
    mean: expectedIt(S0, mu, T),
    variance: varianceIt(S0, mu, sigma, T),
  };
}

// Abramowitz-Stegun 7.1.26 rational approximation of erf; |error| < 1.5e-7.
// Good enough for the §3e first-passage anchor (tests use 4·stderr tolerance
// which dominates for nPaths ≤ 100k).
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

export function standardNormalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

// P[τ ≤ T] for τ = first passage of GBM S_t to H = h·S_0. Reduces to the
// Brownian-motion-with-drift hitting distribution on X_t = log(S_t/S_0):
//   X_t = ν·t + σ·W_t,  ν = μ − σ²/2,  barrier b = log h > 0.
// Harrison (1985) / Borodin-Salminen Table 3.0.1:
//   P(τ ≤ T) = Φ((νT − b)/(σ√T)) + e^{2νb/σ²} · Φ((−νT − b)/(σ√T)).
// h ≤ 1: barrier at or below S_0 ⇒ fires immediately ⇒ return 1.
// σ = 0: deterministic drift, hits iff ν·T ≥ b ⇒ step function.
export function firstPassageProb(
  mu: number,
  sigma: number,
  T: number,
  h: number,
): number {
  if (!(h > 1)) return 1;
  if (!(T > 0)) return 0;
  const b = Math.log(h);
  if (!(sigma > 0)) {
    const nu = mu;
    return nu * T >= b ? 1 : 0;
  }
  const nu = mu - 0.5 * sigma * sigma;
  const sqrtT = Math.sqrt(T);
  const a = (nu * T - b) / (sigma * sqrtT);
  const c = (-nu * T - b) / (sigma * sqrtT);
  const weight = Math.exp((2 * nu * b) / (sigma * sigma));
  return standardNormalCdf(a) + weight * standardNormalCdf(c);
}

// E[τ ∧ T] = ∫₀ᵀ P(τ > t) dt under pure GBM. Evaluated by composite Simpson
// on the CDF from `firstPassageProb`; N=200 subintervals holds the integrand
// to <1e-6 for the (μ, σ, T, h) ranges covered by tests.
export function expectedHittingTime(
  mu: number,
  sigma: number,
  T: number,
  h: number,
  nSubdiv = 200,
): number {
  if (!(T > 0)) return 0;
  if (!(h > 1)) return 0;
  const n = nSubdiv % 2 === 0 ? nSubdiv : nSubdiv + 1;
  const dt = T / n;
  let acc = 0;
  for (let k = 0; k <= n; k++) {
    const t = k * dt;
    const surv = t === 0 ? 1 : 1 - firstPassageProb(mu, sigma, t, h);
    const w = k === 0 || k === n ? 1 : k % 2 === 0 ? 2 : 4;
    acc += w * surv;
  }
  return (acc * dt) / 3;
}
