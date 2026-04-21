// Switching-operating-book Monte Carlo runner. The book's mode is a pure
// state function of the kVCM spot relative to the threshold H = h · S_0:
// principal (b2b) pricing when S_t < H, fee-based pricing at rate f_post
// when S_t ≥ H. The book rides the spot across H symmetrically — it flips
// back to b2b whenever S_t falls below H, and the horizon can contain any
// number of crossings.
//
// This runner emits the **pure operating book** `switching_op`:
//
//   switching_op(T) = Q · λ · T_b2b − P · λ · I_b2b + f_post · P · λ · I_fee,
//
// starting from P&L(0) = 0, where
//   T_b2b   = ∫₀ᵀ 1{S_t < H} dt
//   I_b2b   = ∫₀ᵀ 1{S_t < H} · S_t dt
//   I_fee   = ∫₀ᵀ 1{S_t ≥ H} · S_t dt      (I_b2b + I_fee = I_T).
//
// Desk compositions (switching + treasury, or syndicated switching) are
// assembled at the consumer level from these samples plus the treasury
// samples from `simulate-run`.
//
// The inner log-Euler step is inlined rather than delegated to `samplePath`
// because the mode check wants per-step access to (Sprev, Si) and the
// trapezoid contribution must be routed to the correct bucket (I_b2b vs
// I_fee). RNG consumption per step matches `samplePath` exactly, so runs
// that share `seed` with `simulate()` draw bit-for-bit identical paths.
//
// Pure-GBM closed-form anchors (see ./moments.ts) z-test the simulator on
// every run:
//   P[τ ≤ T]    from `firstPassageProb`
//   E[τ ∧ T]    from `expectedHittingTime`          (τ is a path property)
//   E[T_fee]    from `expectedTimeAboveBarrier`
//   E[I_fee]    from `expectedIntegralAboveBarrier`

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
  /** Fixed USD quote per tonne (b2b-mode pricing). */
  Q: number;
  /** Fee rate used for the same-seed fee-book reference (`fee · P · λ · I_T`)
   *  and for the switching book's fee-mode leg when `feePost` resolves to
   *  `null` (zero-config lock-to-f path). */
  fee: number;
  /** Switching-variant threshold ratio h = H/S0. Infinity disables the switch.
   *  h ≤ 1 starts the book in fee mode (the spot is already at or above H). */
  barrierRatio: number;
  /** Switching-variant fee-mode fee rate. `null` locks it to `fee`. */
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
  /** Time spent in b2b mode per path, ∈ [0, T]. */
  tB2bSamples: Float64Array;
  /** Time spent in fee mode per path, = T − tB2b. */
  tFeeSamples: Float64Array;
  /** 1 iff the path ever entered fee mode (S_t ≥ H at some t). */
  everCrossedMask: Uint8Array;
  /** First time S_t crosses H per path, or T if it never does. Path-level
   *  property independent of the book's response; the Harrison /
   *  Borodin-Salminen expected-hitting-time oracle anchors
   *  `mean(firstCrossTimeSamples)`. */
  firstCrossTimeSamples: Float64Array;
  /** Number of times S_t crossed the threshold H per path (raw count of
   *  sign(S − H) flips along the discretised path). */
  nCrossingsSamples: Uint32Array;
  /** ∫₀ᵀ 1{mode = b2b} · S_t dt per path. */
  IB2bSamples: Float64Array;
  /** ∫₀ᵀ 1{mode = fee} · S_t dt per path. I_b2b + I_fee = I_T exactly by
   *  construction. */
  IFeeSamples: Float64Array;
  ITSamples: Float64Array;
  terminalS: Float64Array;
  sampledPaths: Float64Array[];
  /** Inventory notional N = λ · T. */
  N: number;
  /** Post-switch fee rate resolved at call time (`fee` if feePost = null). */
  feePostResolved: number;
  /** Threshold level H = h · S0, materialised for plot overlays. */
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
    // Path-level first-crossing time τ := inf{t : S_t ≥ H} ∧ T. Starts at
    // T (never-crosses fallback) and is overwritten on the first step
    // that completes above H; h ≤ 1 (Sprev ≥ H at init) gives τ = 0.
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

      // Per-step trapezoid piece ½(S_{k-1} + S_k)·dt. Summing reproduces
      // samplePath's weighting exactly, so I_b2b + I_fee = I_T to machine
      // precision. The piece is bucketed by the entry-mode indicator
      // 1{Sprev ≥ H}; a step that crosses the threshold mid-flight
      // contributes its whole piece to the entry-mode bucket, with an
      // O(dt) discretisation error that vanishes as nSteps → ∞.
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
