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

// E[S_T] under GBM (and under compensated Merton — the compensation
// identity preserves the mean of the price process). Used by the
// treasury closed form whenever k_pre > N·P (over-hedged leftover marked
// at S_T).
export function expectedST(S0: number, mu: number, T: number): number {
  return S0 * Math.exp(mu * T);
}

// Var[S_T] under pure GBM. expm1 keeps precision for small σ²T.
export function varianceST(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): number {
  return S0 * S0 * Math.exp(2 * mu * T) * Math.expm1(sigma * sigma * T);
}

// Cov[I_T, S_T] under pure GBM. Derived from E[S_t·S_T] = S_0²·exp(μ(t+T) + σ²·t)
// for t ≤ T, integrated over [0, T]:
//   E[I_T · S_T] = ∫₀ᵀ S_0²·e^{μ(t+T)}·e^{σ²t} dt = S_0²·e^{μT} · ∫₀ᵀ e^{(μ+σ²)t} dt
// so Cov = S_0²·e^{μT} · T · (expm1OverX((μ+σ²)T) − expm1OverX(μT)).
export function covarITST(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): number {
  const s2 = sigma * sigma;
  return (
    S0 * S0 * Math.exp(mu * T) * T *
    (expm1OverX((mu + s2) * T) - expm1OverX(mu * T))
  );
}

// Abramowitz-Stegun 7.1.26 rational approximation of erf; |error| < 1.5e-7.
// Good enough for the switching-variant first-passage anchor (tests use 4·stderr tolerance
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

// Two-way switching closed-form anchors. Under pure GBM with X_t = log(S_t/S_0)
// ~ N(νt, σ²t) (ν := μ − σ²/2),
//   P[S_t ≥ h·S_0] = Φ((νt − log h) / (σ√t)),
// and the lognormal partial-expectation identity for X ~ N(m, v²),
//   E[e^X · 1{X ≥ c}] = e^{m+v²/2} · Φ((m + v² − c)/v),
// specialised to (m, v²) = (νt, σ²t), c = log h, gives
//   E[S_t · 1{S_t ≥ h·S_0}]
//     = S_0 · e^{μt} · Φ((μt + σ²t/2 − log h) / (σ√t)).
// Integrating both over [0, T] yields the mean-level anchors consumed by
// `report.ts` to z-test the MC E[time in fee mode] and E[I_fee].

// Tail CDF Φ((νt − log h) / (σ√t)): smooth limit at t → 0⁺.
function probAboveBarrier(
  mu: number,
  sigma: number,
  t: number,
  h: number,
): number {
  if (h <= 0) return 1;
  const logh = Math.log(h);
  if (!(t > 0)) return logh > 0 ? 0 : logh < 0 ? 1 : 0.5;
  if (!(sigma > 0)) {
    const nu = mu;
    return nu * t >= logh ? 1 : 0;
  }
  const nu = mu - 0.5 * sigma * sigma;
  return standardNormalCdf((nu * t - logh) / (sigma * Math.sqrt(t)));
}

// E[time in fee mode] = ∫₀ᵀ P[S_t ≥ h·S_0] dt under pure GBM. Composite Simpson,
// same pattern as `expectedHittingTime`.
export function expectedTimeAboveBarrier(
  mu: number,
  sigma: number,
  T: number,
  h: number,
  nSubdiv = 200,
): number {
  if (!(T > 0)) return 0;
  if (h <= 0) return T;
  const n = nSubdiv % 2 === 0 ? nSubdiv : nSubdiv + 1;
  const dt = T / n;
  let acc = 0;
  for (let k = 0; k <= n; k++) {
    const t = k * dt;
    const p = probAboveBarrier(mu, sigma, t, h);
    const w = k === 0 || k === n ? 1 : k % 2 === 0 ? 2 : 4;
    acc += w * p;
  }
  return (acc * dt) / 3;
}

// E[I_fee] = ∫₀ᵀ E[S_t · 1{S_t ≥ h·S_0}] dt under pure GBM. Composite Simpson.
// h ≤ 0 reduces to the ordinary E[I_T] (barrier is never hit from above).
export function expectedIntegralAboveBarrier(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
  h: number,
  nSubdiv = 200,
): number {
  if (!(T > 0)) return 0;
  if (h <= 0) return expectedIt(S0, mu, T);
  const logh = Math.log(h);
  const n = nSubdiv % 2 === 0 ? nSubdiv : nSubdiv + 1;
  const dt = T / n;
  let acc = 0;
  for (let k = 0; k <= n; k++) {
    const t = k * dt;
    const integrand = (() => {
      if (!(t > 0)) {
        // t = 0 limit: S_0 · 1{S_0 ≥ h·S_0} = S_0 · 1{h ≤ 1}; strict inequality
        // goes to Φ(∓∞) inside the formula, so use the indicator directly.
        return h <= 1 ? S0 : 0;
      }
      if (!(sigma > 0)) {
        // Deterministic: S_t = S_0 · e^{μt}, integrand = S_t · 1{e^{μt} ≥ h}.
        const St = S0 * Math.exp(mu * t);
        return mu * t >= logh ? St : 0;
      }
      const d = (mu * t + 0.5 * sigma * sigma * t - logh) / (sigma * Math.sqrt(t));
      return S0 * Math.exp(mu * t) * standardNormalCdf(d);
    })();
    const w = k === 0 || k === n ? 1 : k % 2 === 0 ? 2 : 4;
    acc += w * integrand;
  }
  return (acc * dt) / 3;
}
