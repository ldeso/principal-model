// Switching operating book MC runner. Mode: fee if S_t ≥ H = h·S_0, b2b
// otherwise; indicator rides S_t symmetrically.
//
//   switching_op(T) = Q·λ·T_b2b − P·λ·I_b2b + f_post·P·λ·I_fee,
//
// starting at P&L(0) = 0. I_b2b + I_fee = I_T by construction.
//
// The inner log-Euler step is inlined to bucket per-step trapezoid pieces
// into I_b2b / I_fee by the entry-mode indicator 1{Sprev ≥ H}. RNG
// consumption matches `samplePath`, so a shared seed draws identical paths.
//
// Pure-GBM anchors (see ./moments.ts):
//   P[τ ≤ T]    firstPassageProb
//   E[τ ∧ T]    expectedHittingTime
//   E[T_fee]    expectedTimeAboveBarrier
//   E[I_fee]    expectedIntegralAboveBarrier

import { mulberry32 } from "./rng.js";

export interface SwitchingInputs {
  S0: number;
  mu: number;
  sigma: number;
  P: number;
  lambda: number;
  T: number;
  Q: number;
  /** Fee rate; `feePost = null` locks the fee-mode rate to this. */
  fee: number;
  /** Threshold ratio; Infinity disables; h ≤ 1 starts in fee mode. */
  barrierRatio: number;
  /** Fee-mode rate; `null` ⇒ locked to `fee`. */
  feePost: number | null;
  /** 0 ⇒ pure GBM. */
  lambdaJ?: number;
  muJ?: number;
  sigmaJ?: number;
  nPaths: number;
  nSteps: number;
  seed: number;
  /** Sample paths retained; capped by nPaths. */
  keepPaths?: number;
}

export interface SwitchingResult {
  /** Pure switching operating book. */
  pnlSamples: Float64Array;
  /** Same-seed b2b reference: Q·N − P·λ·I_T. */
  b2bSamples: Float64Array;
  /** Same-seed fee reference: fee·P·λ·I_T. */
  feeSamples: Float64Array;
  /** Time in b2b mode per path. */
  tB2bSamples: Float64Array;
  /** T − tB2b. */
  tFeeSamples: Float64Array;
  /** 1 iff the path entered fee mode. */
  everCrossedMask: Uint8Array;
  /** First S_t ≥ H time per path, or T. */
  firstCrossTimeSamples: Float64Array;
  /** Number of discrete sign(S − H) flips per path. */
  nCrossingsSamples: Uint32Array;
  IB2bSamples: Float64Array;
  IFeeSamples: Float64Array;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** N = λ·T. */
  N: number;
  /** Resolved fee-mode rate. */
  feePostResolved: number;
  /** H = h·S_0. */
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

  // Matches gbm.ts::samplePath so shared-seed draws coincide.
  const kappa = lambdaJ > 0
    ? Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1
    : 0;
  const drift = (mu - 0.5 * sigma * sigma - lambdaJ * kappa) * dt;
  const diffusion = sigma * Math.sqrt(dt);
  const lamDt = lambdaJ * dt;

  const pnlSamples = new Float64Array(nPaths);
  const b2bSamples = new Float64Array(nPaths);
  const feeSamples = new Float64Array(nPaths);
  const tB2bSamples = new Float64Array(nPaths);
  const tFeeSamples = new Float64Array(nPaths);
  const everCrossedMask = new Uint8Array(nPaths);
  const firstCrossTimeSamples = new Float64Array(nPaths);
  const nCrossingsSamples = new Uint32Array(nPaths);
  const IB2bSamples = new Float64Array(nPaths);
  const IFeeSamples = new Float64Array(nPaths);
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
    let IB2b = 0;
    let IFee = 0;
    let tB2b = 0;
    let tFee = 0;
    let everCrossed = Sprev >= H;
    // τ := inf{t : S_t ≥ H} ∧ T; h ≤ 1 (Sprev ≥ H) ⇒ τ = 0.
    let firstCrossTime = everCrossed ? 0 : T;
    let crossings = 0;

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

      // Bucket ½(S_{k-1} + S_k)·dt by the entry-mode indicator.
      // Sum of pieces reproduces samplePath's I_T to machine precision.
      const piece = 0.5 * (Sprev + Si) * dt;
      const prevAbove = Sprev >= H;

      if (prevAbove) {
        IFee += piece;
        tFee += dt;
      } else {
        IB2b += piece;
        tB2b += dt;
      }

      const currAbove = Si >= H;
      if (prevAbove !== currAbove) crossings += 1;
      if (currAbove && !everCrossed) {
        firstCrossTime = k * dt;
        everCrossed = true;
      }

      Sprev = Si;
    }

    const IT = IB2b + IFee;
    terminalS[i] = Sprev;
    ITSamples[i] = IT;
    IB2bSamples[i] = IB2b;
    IFeeSamples[i] = IFee;
    tB2bSamples[i] = tB2b;
    tFeeSamples[i] = tFee;
    nCrossingsSamples[i] = crossings;
    firstCrossTimeSamples[i] = firstCrossTime;
    everCrossedMask[i] = everCrossed ? 1 : 0;

    feeSamples[i] = fee * P * lambda * IT;
    b2bSamples[i] = Q * N - P * lambda * IT;
    pnlSamples[i] =
      Q * lambda * tB2b - P * lambda * IB2b + feePostResolved * P * lambda * IFee;

    if (Spath) sampledPaths.push(Spath);
  }

  return {
    pnlSamples,
    b2bSamples,
    feeSamples,
    tB2bSamples,
    tFeeSamples,
    everCrossedMask,
    firstCrossTimeSamples,
    nCrossingsSamples,
    IB2bSamples,
    IFeeSamples,
    ITSamples,
    terminalS,
    sampledPaths,
    N,
    feePostResolved,
    barrierLevel: H,
  };
}
