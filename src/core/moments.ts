// Closed-form moments of I_T := ∫₀ᵀ S_t dt (Dufresne 2001) and
// barrier / first-passage anchors for the switching book (Harrison 1985;
// Borodin-Salminen).

export interface GbmMoments {
  mean: number;
  variance: number;
}

// (e^x − 1) / x with analytic limit 1 at x = 0.
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
    // μ ≈ −σ² with σ > 0: L'Hôpital in σ² at fixed μ, evaluated at b = μ.
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

export function expectedST(S0: number, mu: number, T: number): number {
  return S0 * Math.exp(mu * T);
}

export function varianceST(
  S0: number,
  mu: number,
  sigma: number,
  T: number,
): number {
  return S0 * S0 * Math.exp(2 * mu * T) * Math.expm1(sigma * sigma * T);
}

// Cov[I_T, S_T] = S₀²·e^{μT}·T·(expm1OverX((μ+σ²)T) − expm1OverX(μT)).
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

// Abramowitz-Stegun 7.1.26, |error| < 1.5e-7.
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

// P[τ ≤ T] for τ = first passage to H = h·S₀; Harrison / Borodin-Salminen.
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

// E[τ ∧ T] = ∫₀ᵀ P(τ > t) dt; composite Simpson, N = 200 subintervals.
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

// P[S_t ≥ h·S₀] = Φ((νt − log h)/(σ√t)) with ν = μ − σ²/2.
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

// E[T_fee] = ∫₀ᵀ P[S_t ≥ h·S₀] dt; composite Simpson.
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

// E[I_fee] = ∫₀ᵀ E[S_t·1{S_t ≥ h·S₀}] dt; composite Simpson.
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
      if (!(t > 0)) return h <= 1 ? S0 : 0;
      if (!(sigma > 0)) {
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
