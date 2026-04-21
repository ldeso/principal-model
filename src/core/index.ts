// OJS-facing barrel. The Simulator's OJS imports from the compiled form at
// report/lib/compiled/src/core/index.js.

export { mulberry32 } from "./rng.js";
export type { Rng } from "./rng.js";
export { samplePath } from "./gbm.js";
export type { GbmPath, SamplePathOpts } from "./gbm.js";
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
  formatTickCurrency,
  formatTickDate,
  tickStep,
  xTicksAnchoredRight,
  xTicksForHorizon,
} from "./ticks.js";
