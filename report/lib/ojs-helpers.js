// Shared helpers for OJS cells in phase-c.qmd and conclusions.qmd.
// Imported via: import {...} from "./lib/ojs-helpers.js"

export function makeRng(seedVal) {
  let state = seedVal >>> 0;
  let cached = null;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return u === 0 ? 1 / 4294967296 : u;
  };
  const normal = () => {
    if (cached !== null) {
      const z = cached;
      cached = null;
      return z;
    }
    const u1 = uniform();
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const th = 2 * Math.PI * u2;
    cached = r * Math.sin(th);
    return r * Math.cos(th);
  };
  // Knuth's product-of-uniforms Poisson sampler. Always called here with
  // lambda = λ_J · dt ≪ 1, so the loop resolves in one or two iterations.
  const poisson = (lambda) => {
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

export function expm1OverX(x) {
  if (x === 0) return 1;
  if (Math.abs(x) < 1e-8) return 1 + x / 2 + (x * x) / 6;
  return Math.expm1(x) / x;
}

export function quantileSorted(sorted, q) {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

export function tailMean(sortedAscending, q) {
  const n = sortedAscending.length;
  const k = Math.max(1, Math.ceil((1 - q) * n));
  let s = 0;
  for (let i = 0; i < k; i++) s += sortedAscending[i];
  return s / k;
}

// Calendar-ish tick step in days — whole weeks up to 60 days, whole months
// up to 2 years, then whole years. Shared by the two tick functions and by
// formatTickDate so the three always agree.
export function tickStep(tdays) {
  if (tdays <= 60) return 7;
  if (tdays <= 730) return 30;
  return 365;
}

// Pick calendar-ish tick positions (in days) so the x-axis steps in
// whole weeks / months / years depending on the horizon, rather than
// rescaling to a fixed number of evenly-spaced ticks.
export function xTicksForHorizon(tdays) {
  const step = tickStep(tdays);
  const ticks = [];
  for (let d = 0; d <= tdays; d += step) ticks.push(d);
  return ticks;
}

// Same step logic as xTicksForHorizon but anchored on the right edge
// (day = tdays) rather than zero.
export function xTicksAnchoredRight(tdays) {
  const step = tickStep(tdays);
  const ticks = [];
  for (let d = tdays; d >= 0; d -= step) ticks.push(d);
  return ticks.reverse();
}

// Render a date at a resolution appropriate for the horizon it sits on.
export function formatTickDate(date, tdays) {
  const step = tickStep(tdays);
  if (step === 7) {
    return date.toLocaleString("en-US", { month: "short", day: "numeric" });
  }
  if (step === 30) {
    return date.toLocaleString("en-US", { month: "short", year: "2-digit" });
  }
  return date.toLocaleString("en-US", { year: "numeric" });
}

// Monte Carlo simulator. Merton jump-diffusion when lambdaJ > 0, otherwise
// pure GBM — the jump branch is skipped entirely and the compensator term
// vanishes, so the output is numerically identical to the GBM-only form.
// Pass keepPaths > 0 to retain up to that many sample trajectories for
// rendering.
export function simulate(inputs) {
  const {
    S0, mu, sigma, P, lambda, T, Q, fee, kPre, cBasis,
    lambdaJ = 0, muJ = 0, sigmaJ = 0,
    nPaths, nSteps, seed,
    keepPaths = 0,
  } = inputs;
  const rng = makeRng(seed);
  const dt = T / nSteps;
  // Compensated Merton drift: subtracting λ_J·κ keeps E[S_t] = S_0·e^{μt}
  // regardless of jump settings, so turning jumps on widens the
  // distributions without shifting their means.
  const kappa = lambdaJ > 0
    ? Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1
    : 0;
  const drift = (mu - 0.5 * sigma * sigma - lambdaJ * kappa) * dt;
  const diffusion = sigma * Math.sqrt(dt);
  const lamDt = lambdaJ * dt;
  const N = lambda * T;
  // Fraction of horizon over which inventory lasts, clamped to [0, 1].
  // Retirements are a constant flow λ, so `kPre` tokens cover
  // τ = kPre / (λ P) years of demand.
  const tauFrac = (lambda > 0 && P > 0)
    ? Math.min(1, kPre / (lambda * P * T))
    : 1;
  const tokensUsedInternal = Math.min(kPre, lambda * T * P);
  const tokensLeftover = Math.max(0, kPre - lambda * T * P);
  // Snap tauFrac onto the integration grid and treat "only the endpoint
  // falls in [τT, T]" as an empty range — otherwise the trapezoid rule
  // would contribute 0.5·S_T·dt for a zero-length interval.
  const tailStartRaw = Math.ceil(tauFrac * nSteps);
  const tailStartStep = tailStartRaw >= nSteps ? nSteps + 1 : tailStartRaw;

  const feeSamples = new Float64Array(nPaths);
  const principalSamples = new Float64Array(nPaths);
  const b2bSamples = new Float64Array(nPaths);
  const ITSamples = new Float64Array(nPaths);
  const terminalS = new Float64Array(nPaths);
  const keep = Math.min(keepPaths, nPaths);
  const sampledPaths = [];

  for (let i = 0; i < nPaths; i++) {
    // Trapezoid accumulators: full-horizon I_T and uncovered-tail integral
    // ∫_{τ T}^{T} S_t dt. We stitch the tail integral out of the same
    // trajectory so it uses the same realisations as I_T.
    let S = S0;
    let IT = 0;
    let tailInt = 0;
    const path = (i < keep) ? new Float64Array(nSteps + 1) : null;
    if (path) path[0] = S;

    // Trapezoid rule: weight endpoints by 1/2.
    IT += 0.5 * S;
    if (tailStartStep === 0) tailInt += 0.5 * S;

    for (let k = 1; k <= nSteps; k++) {
      const z = rng.normal();
      let jumpSum = 0;
      if (lambdaJ > 0) {
        const nJ = rng.poisson(lamDt);
        for (let j = 0; j < nJ; j++) {
          jumpSum += muJ + sigmaJ * rng.normal();
        }
      }
      S = S * Math.exp(drift + diffusion * z + jumpSum);
      if (path) path[k] = S;
      const w = (k === nSteps) ? 0.5 : 1;
      IT += w * S;
      if (k >= tailStartStep) {
        const wt = (k === tailStartStep || k === nSteps) ? 0.5 : 1;
        tailInt += wt * S;
      }
    }
    IT *= dt;
    tailInt *= dt;
    terminalS[i] = S;
    ITSamples[i] = IT;

    // Fee book: R_fee = f · P · λ · I_T.
    feeSamples[i] = fee * P * lambda * IT;

    // Back-to-back principal book (no pre-purchase): Π = Q·N − P·λ·I_T.
    b2bSamples[i] = Q * N - P * lambda * IT;

    // Custom principal with initial inventory:
    //   Π = Q·N − C_basis − P·λ·∫_{τT}^{T} S_t dt + leftover · S_T.
    // The cost basis is a sunk cost at t=0; uncovered retirements after
    // inventory exhaustion are bought at spot; excess tokens mark-to-market
    // at horizon.
    principalSamples[i] =
      Q * N - cBasis - P * lambda * tailInt + tokensLeftover * S;

    if (path) sampledPaths.push(path);
  }

  return {
    feeSamples,
    principalSamples,
    b2bSamples,
    ITSamples,
    terminalS,
    sampledPaths,
    tauFrac,
    tokensUsedInternal,
    tokensLeftover,
    N,
  };
}

// Summarise a 1-D sample array: mean, SD, 95% CI, historical VaR/CVaR,
// loss probability, Sharpe.
export function summarise(samples) {
  const n = samples.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i];
  const mean = sum / n;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean;
    sse += d * d;
  }
  const variance = n > 1 ? sse / (n - 1) : 0;
  const sd = Math.sqrt(variance);
  const stderr = sd / Math.sqrt(n);
  const sorted = Float64Array.from(samples); sorted.sort();
  const var95 = -quantileSorted(sorted, 0.05);
  const var99 = -quantileSorted(sorted, 0.01);
  const cvar95 = -tailMean(sorted, 0.95);
  const cvar99 = -tailMean(sorted, 0.99);
  let losses = 0;
  for (let i = 0; i < n; i++) if (samples[i] < 0) losses++;
  const probLoss = losses / n;
  const sharpe = sd > 0 ? mean / sd : null;
  return { mean, sd, stderr, ci95: 1.96 * stderr, var95, var99, cvar95, cvar99, probLoss, sharpe };
}
