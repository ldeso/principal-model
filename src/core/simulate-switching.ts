// Switching-operating-book Monte Carlo runner. The book starts in b2b mode and
// flips to fee-based pricing at rate `feePost` the first time `S_t` crosses
// `h · S_0`; τ is the stopping time. No closed-form density (τ is not a path
// integral), so MC is authoritative. Two pure-GBM anchors survive — P[τ ≤ T]
// and E[τ ∧ T] — and live in ./moments.ts (`firstPassageProb`,
// `expectedHittingTime`).
//
// This runner emits the **pure operating book** `switching_op`:
//
//   switching_op(T) = Q · λ · τ − P · λ · I_τ + f_post · P · λ · J_τ,
//
// starting from P&L(0) = 0. Desk compositions (switching + treasury, or
// syndicated switching) are assembled at the consumer level from these samples
// plus the treasury samples from `simulate-run`.
//
// The inner log-Euler step is inlined rather than delegated to `samplePath`
// because the barrier check wants per-step access to (Sprev, Si) and the
// trapezoid contribution must be routed to the correct bucket (I_τ or J_τ).
// RNG consumption per step matches `samplePath` exactly, so runs that share
// `seed` with `simulate()` draw bit-for-bit identical paths.

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
  /** Fee rate applied to pre-switch fee-book revenue AND to J_τ when
   *  `feePost` resolves to `null`. */
  fee: number;
  /** Switching-variant barrier ratio h = H/S0. Infinity disables the switch.
   *  h ≤ 1 fires the switch immediately at t = 0. */
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
  /** Pure switching operating book. */
  pnlSamples: Float64Array;
  /** Same-seed b2b operating reference: Q · N − P · λ · I_T. */
  b2bSamples: Float64Array;
  /** Same-seed fee operating reference: fee · P · λ · I_T. */
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
  /** Inventory notional N = λ · T. */
  N: number;
  /** Post-switch fee rate resolved at call time (`fee` if feePost = null). */
  feePostResolved: number;
  /** Barrier level H = h · S0, materialised for plot overlays. */
  barrierLevel: number;
}

export function simulateSwitching(inputs: SwitchingInputs): SwitchingResult {
  const {
    S0, mu, sigma, P, lambda, T, Q, fee,
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

      // Per-step trapezoid piece ½(S_{k-1} + S_k)·dt. Summing reproduces
      // samplePath's weighting exactly, so I_τ + J_τ = I_T to machine
      // precision. The step that completes the crossing belongs to the
      // pre-switch bucket (matches simulate-run.ts' tailStartStep rule).
      const piece = 0.5 * (Sprev + Si) * dt;

      if (!switched && Si >= H) {
        tau = k * dt;
        switched = true;
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
    pnlSamples[i] =
      Q * lambda * tau - P * lambda * ITau + feePostResolved * P * lambda * JTau;

    if (Spath) sampledPaths.push(Spath);
  }

  return {
    pnlSamples,
    b2bSamples,
    feeSamples,
    tauSamples,
    switchedMask,
    ITauSamples,
    JTauSamples,
    ITSamples,
    terminalS,
    sampledPaths,
    N,
    feePostResolved,
    barrierLevel: H,
  };
}
