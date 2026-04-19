// Barrel export for the OJS-facing API. Phase C cells import from the
// compiled form of this file at report/lib/compiled/src/core/index.js so
// the browser runs the same code the Vitest suite covers.

export { mulberry32 } from "./rng.js";
export type { Rng } from "./rng.js";
export {
  expectedHittingTime,
  expm1OverX,
  firstPassageProb,
  standardNormalCdf,
} from "./moments.js";
export { summarise } from "./risk.js";
export type { FullSummary } from "./risk.js";
export { simulateRun as simulate } from "./simulate-run.js";
export type {
  SimulateRunInputs,
  SimulateRunResult,
} from "./simulate-run.js";
export { simulateSwitching } from "./simulate-switching.js";
export type {
  SwitchingInputs,
  SwitchingResult,
} from "./simulate-switching.js";
export {
  formatTickDate,
  tickStep,
  xTicksAnchoredRight,
  xTicksForHorizon,
} from "./ticks.js";
