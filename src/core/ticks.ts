// Horizon-axis tick helpers for OJS plots.

export function tickStep(tdays: number): number {
  if (tdays <= 60) return 7;
  if (tdays <= 730) return 30;
  return 365;
}

export function xTicksForHorizon(tdays: number): number[] {
  const step = tickStep(tdays);
  const ticks: number[] = [];
  for (let d = 0; d <= tdays; d += step) ticks.push(d);
  return ticks;
}

export function xTicksAnchoredRight(tdays: number): number[] {
  const step = tickStep(tdays);
  const ticks: number[] = [];
  for (let d = tdays; d >= 0; d -= step) ticks.push(d);
  return ticks.reverse();
}

export function formatTickDate(date: Date, tdays: number): string {
  const step = tickStep(tdays);
  if (step === 7) {
    return date.toLocaleString("en-US", { month: "short", day: "numeric" });
  }
  if (step === 30) {
    return date.toLocaleString("en-US", { month: "short", year: "2-digit" });
  }
  return date.toLocaleString("en-US", { year: "numeric" });
}

// Currency formatter matching d3's "$~s" for |v| ≥ 1000; for |v| < 1 uses
// decimal notation rather than d3's milli prefix.
export function formatTickCurrency(v: number): string {
  if (!Number.isFinite(v)) return "";
  if (v === 0) return "$0";
  const sign = v < 0 ? "-" : "";
  const x = Math.abs(v);
  const suffixes = ["", "k", "M", "B", "T"] as const;
  let tier = Math.min(
    suffixes.length - 1,
    Math.max(0, Math.floor(Math.log10(x) / 3)),
  );
  let scaled = x / Math.pow(1000, tier);
  // Rounding can spill across a tier boundary (999.9 → "1.00e+3").
  if (scaled >= 999.5 && tier < suffixes.length - 1) {
    tier += 1;
    scaled = x / Math.pow(1000, tier);
  }
  const body = Number.parseFloat(scaled.toPrecision(3)).toString();
  return `${sign}$${body}${suffixes[tier]}`;
}
