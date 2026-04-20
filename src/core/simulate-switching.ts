// Switching-strategy Monte Carlo runner. Start in principal back-to-back mode,
// flip to fee-based the first time S_t ≥ h·S0, compose with the partial variant (α)
// and the syndicated variant (β, θ, mode). No closed form for the P&L density: τ is a stopping time so
// the distribution depends on path geometry rather than I_T alone. Two pure-
// GBM anchors survive for validation: P[τ ≤ T] and E[τ ∧ T] live in
// ./moments.ts (`firstPassageProb`, `expectedHittingTime`).
//
// The inner log-Euler step is inlined rather than delegated to `samplePath`
// because the barrier check wants per-step access to (Sprev, Si) and the
// trapezoid contribution must be routed to the correct bucket (I_τ or J_τ).
// RNG consumption per step matches `samplePath` exactly — one `rng.normal()`
// plus an optional Poisson jump block — so runs that share `seed` with
// `simulate()` draw bit-for-bit identical paths.

import { mulberry32 } from "./rng.js";

export interface SwitchingInputs {
  S0: number;
  mu: number;
  sigma: number;
  /** Protocol price kVCM/tonne. */
  P: number;
  /** Retirement flow, tonnes per unit time. */
  lambda: number;
  T: number;
  /** Fixed USD quote per tonne (pre-switch). */
  Q: number;
  /** Fee rate applied to pre-switch fee-book revenue AND to J_τ when `feePost`
   *  resolves to `null`. */
  fee: number;
  /** Pre-purchase fraction for the partial variant. */
  alpha: number;
  /** Syndicated-variant external cession fraction. Default 0. */
  beta?: number;
  /** Syndicated-variant counterparty risk-load multiplier θ ≥ 0. Default 0 ⇒ fair premium. */
  premiumLoad?: number;
  /** Syndicated-variant risk-measure basis for the load. Default `"sharpe"`. */
  premiumMode?: "sharpe" | "cvar";
  /** Switching-variant barrier ratio h = H/S0. Infinity disables the switch (returns a
   *  syndicated-variant-equivalent run). h ≤ 1 fires the switch immediately at t = 0. */
  barrierRatio: number;
  /** Switching-variant post-switch fee rate. `null` locks it to `fee`. */
  feePost: number | null;
  /** Merton jump intensity. 0 ⇒ pure GBM. */
  lambdaJ?: number;
  muJ?: number;
  sigmaJ?: number;
  nPaths: number;
  nSteps: number;
  seed: number;
  /** Sample trajectories retained for plotting; capped by nPaths. */
  keepPaths?: number;
}

export interface SwitchingResult {
  /** Composed P&L Π_{3e} = α·N·(Q − P·S0) + (1−α)(1−β)·Π_sw + π_loaded. */
  pnlSamples: Float64Array;
  /** Per-path stochastic leg Π_sw^{(1-α)} = Q·λ·τ − P·λ·I_τ + f_post·P·λ·J_τ
   *  (before composition with α, β). Exposed so the Validation page and
   *  tests can read the switching formula directly without re-deriving it
   *  from pnlSamples. */
  stochLegSamples: Float64Array;
  /** Same-seed back-to-back reference: Q·N − P·λ·I_T. */
  b2bSamples: Float64Array;
  /** Same-seed fee-book reference: fee·P·λ·I_T (uses `fee`, not `feePost`). */
  feeSamples: Float64Array;
  /** Stopping time ∈ [0, T]; = T on paths that never cross the barrier. */
  tauSamples: Float64Array;
  /** 1 iff τ < T, i.e. the path switched. */
  switchedMask: Uint8Array;
  /** ∫₀^τ S_t dt per path. */
  ITauSamples: Float64Array;
  /** ∫_τ^T S_t dt per path. I_τ + J_τ = I_T exactly by construction. */
  JTauSamples: Float64Array;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** MC-derived fair premium: β·(1−α)·E[Π_sw^{(1-α)}]. */
  premiumFair: number;
  /** MC-derived loaded premium actually applied to `pnlSamples`. */
  premiumLoaded: number;
  /** Inventory notional N = λ · T. */
  N: number;
  /** Post-switch fee rate resolved at call time (`fee` if feePost = null). */
  feePostResolved: number;
  /** Barrier level H = h · S0, materialised for plot overlays. */
  barrierLevel: number;
}

// Syndicated-variant Gaussian CVaR95 factor = φ(Φ^{-1}(0.95))/0.05; proxy used in `"cvar"` mode.
// Same constant as in simulate-run.ts and models.ts — kept co-located so the
// CVaR proxy story lives beside the code that uses it.
const GAUSSIAN_CVAR95_FACTOR = 2.062713055949736;

export function simulateSwitching(inputs: SwitchingInputs): SwitchingResult {
  const {
    S0, mu, sigma, P, lambda, T, Q, fee, alpha,
    beta = 0, premiumLoad = 0, premiumMode = "sharpe",
    barrierRatio, feePost,
    lambdaJ = 0, muJ = 0, sigmaJ = 0,
    nPaths, nSteps, seed,
    keepPaths = 0,
  } = inputs;

  const rng = mulberry32(seed);
  const dt = T / nSteps;
  const N = lambda * T;
  const feePostResolved = feePost ?? fee;
  const H = barrierRatio * S0;

  // Jump-compensation κ = E[e^Y − 1]: zero when λ_J = 0, same formula as
  // gbm.ts::samplePath so path draws coincide bit-for-bit under shared seed.
  const kappa = lambdaJ > 0
    ? Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1
    : 0;
  const drift = (mu - 0.5 * sigma * sigma - lambdaJ * kappa) * dt;
  const diffusion = sigma * Math.sqrt(dt);
  const lamDt = lambdaJ * dt;

  const pnlSamples = new Float64Array(nPaths);
  const stochLegSamples = new Float64Array(nPaths);
  const b2bSamples = new Float64Array(nPaths);
  const feeSamples = new Float64Array(nPaths);
  const tauSamples = new Float64Array(nPaths);
  const switchedMask = new Uint8Array(nPaths);
  const ITauSamples = new Float64Array(nPaths);
  const JTauSamples = new Float64Array(nPaths);
  const ITSamples = new Float64Array(nPaths);
  const terminalS = new Float64Array(nPaths);

  const keep = Math.min(keepPaths, nPaths);
  const sampledPaths: Float64Array[] = [];

  const matchedShift = alpha * N * (Q - P * S0);
  const retainedStochScale = (1 - alpha) * (1 - beta);
  // Cession fraction applied when computing the MC-derived premium scalars.
  const cession = beta * (1 - alpha);

  for (let i = 0; i < nPaths; i++) {
    const storeFull = i < keep;
    const Spath: Float64Array | null = storeFull
      ? new Float64Array(nSteps + 1)
      : null;
    if (Spath) Spath[0] = S0;

    let Sprev = S0;
    let ITau = 0;
    let JTau = 0;
    let switched = barrierRatio <= 1;  // h ≤ 1 fires at t=0, no pre-switch bucket
    let tau = switched ? 0 : T;

    for (let k = 1; k <= nSteps; k++) {
      const z = rng.normal();
      let jumpSum = 0;
      if (lambdaJ > 0) {
        const jumps = rng.poisson(lamDt);
        for (let j = 0; j < jumps; j++) {
          jumpSum += muJ + sigmaJ * rng.normal();
        }
      }
      const Si = Sprev * Math.exp(drift + diffusion * z + jumpSum);
      if (Spath) Spath[k] = Si;

      // Per-step trapezoid piece ½(S_{k-1} + S_k)·dt. Summing all pieces
      // reproduces samplePath's (½, 1, …, 1, ½)·dt weighting exactly, so
      // I_τ + J_τ = I_T holds to machine precision.
      const piece = 0.5 * (Sprev + Si) * dt;

      if (!switched && Si >= H) {
        tau = k * dt;
        switched = true;
        // Convention: the step that completed the crossing belongs to the
        // pre-switch bucket (matches simulate-run.ts' tailStartStep rule).
        ITau += piece;
      } else if (!switched) {
        ITau += piece;
      } else {
        JTau += piece;
      }

      Sprev = Si;
    }

    const IT = ITau + JTau;
    terminalS[i] = Sprev;
    ITSamples[i] = IT;
    ITauSamples[i] = ITau;
    JTauSamples[i] = JTau;
    tauSamples[i] = tau;
    switchedMask[i] = tau < T ? 1 : 0;

    feeSamples[i] = fee * P * lambda * IT;
    b2bSamples[i] = Q * N - P * lambda * IT;
    const stochLeg =
      Q * lambda * tau - P * lambda * ITau + feePostResolved * P * lambda * JTau;
    stochLegSamples[i] = stochLeg;

    if (Spath) sampledPaths.push(Spath);
  }

  // Premium derived from MC moments of the stochastic leg. This is the
  // same closure as simulate-run.ts:134-149 but applied to Π_sw^{(1-α)}
  // rather than to a custom-inventory book. Under λ_J = 0, the first two
  // moments have no nice closed form even for the stochastic leg, so MC is
  // authoritative here (unlike the partial/syndicated variants which pipe through Dufresne).
  let stochMean = 0;
  for (let i = 0; i < nPaths; i++) stochMean += stochLegSamples[i] as number;
  stochMean /= nPaths;
  let stochSse = 0;
  for (let i = 0; i < nPaths; i++) {
    const d = (stochLegSamples[i] as number) - stochMean;
    stochSse += d * d;
  }
  const stochVar = nPaths > 1 ? stochSse / (nPaths - 1) : 0;
  const stochSd = Math.sqrt(stochVar);
  const loadFactor = premiumMode === "cvar" ? GAUSSIAN_CVAR95_FACTOR : 1;
  const premiumFair = cession * stochMean;
  const premiumLoaded = premiumFair - cession * premiumLoad * loadFactor * stochSd;

  for (let i = 0; i < nPaths; i++) {
    pnlSamples[i] =
      matchedShift + retainedStochScale * (stochLegSamples[i] as number) + premiumLoaded;
  }

  return {
    pnlSamples,
    stochLegSamples,
    b2bSamples,
    feeSamples,
    tauSamples,
    switchedMask,
    ITauSamples,
    JTauSamples,
    ITSamples,
    terminalS,
    sampledPaths,
    premiumFair,
    premiumLoaded,
    N,
    feePostResolved,
    barrierLevel: H,
  };
}
