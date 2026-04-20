import { describe, expect, it } from "vitest";
import {
  formatTickCurrency,
  formatTickDate,
  tickStep,
  xTicksAnchoredRight,
  xTicksForHorizon,
} from "../src/core/ticks.js";

describe("tickStep", () => {
  it("uses a 7-day step up to and including 60 days", () => {
    expect(tickStep(1)).toBe(7);
    expect(tickStep(30)).toBe(7);
    expect(tickStep(60)).toBe(7);
  });

  it("uses a 30-day step from 61 through 730 days", () => {
    expect(tickStep(61)).toBe(30);
    expect(tickStep(365)).toBe(30);
    expect(tickStep(730)).toBe(30);
  });

  it("uses a 365-day step beyond 730 days", () => {
    expect(tickStep(731)).toBe(365);
    expect(tickStep(3650)).toBe(365);
  });
});

describe("xTicksForHorizon", () => {
  it("starts at 0 and walks forward by tickStep(tdays)", () => {
    for (const t of [14, 60, 180, 365, 1200]) {
      const ticks = xTicksForHorizon(t);
      const step = tickStep(t);
      expect(ticks[0]).toBe(0);
      for (let i = 1; i < ticks.length; i++) {
        expect((ticks[i] as number) - (ticks[i - 1] as number)).toBe(step);
      }
      const last = ticks[ticks.length - 1] as number;
      expect(last).toBeLessThanOrEqual(t);
      expect(last + step).toBeGreaterThan(t);
    }
  });
});

describe("xTicksAnchoredRight", () => {
  it("ends at tdays and walks backward by tickStep(tdays)", () => {
    for (const t of [14, 60, 365, 1200]) {
      const ticks = xTicksAnchoredRight(t);
      const step = tickStep(t);
      expect(ticks[ticks.length - 1]).toBe(t);
      for (let i = 1; i < ticks.length; i++) {
        expect((ticks[i] as number) - (ticks[i - 1] as number)).toBe(step);
      }
      expect(ticks[0] as number).toBeGreaterThanOrEqual(0);
      expect((ticks[0] as number) - step).toBeLessThan(0);
    }
  });

  it("mirrors xTicksForHorizon when tdays is a multiple of the step", () => {
    // One horizon per bucket: 14 (week-step), 180 (month-step),
    // 1095 (year-step); each is a whole number of steps.
    for (const t of [14, 180, 1095]) {
      expect(t % tickStep(t)).toBe(0);
      expect(xTicksAnchoredRight(t)).toEqual(xTicksForHorizon(t));
    }
  });
});

describe("formatTickDate", () => {
  // Local-time constructor + default-timezone toLocaleString ⇒ the calendar
  // day rendered is independent of whichever TZ the test host runs in.
  const d = new Date(2026, 3, 19, 12, 0, 0); // April 19 2026, local noon

  it("renders month-and-day for weekly horizons", () => {
    expect(formatTickDate(d, 30)).toBe("Apr 19");
  });

  it("renders month-and-2-digit-year for monthly horizons", () => {
    expect(formatTickDate(d, 365)).toBe("Apr 26");
  });

  it("renders the 4-digit year for multi-year horizons", () => {
    expect(formatTickDate(d, 2000)).toBe("2026");
  });
});

describe("formatTickCurrency", () => {
  it("renders zero as a plain sigil", () => {
    expect(formatTickCurrency(0)).toBe("$0");
  });

  it("uses decimal notation for |v| < 1 (never the milli prefix)", () => {
    expect(formatTickCurrency(0.1)).toBe("$0.1");
    expect(formatTickCurrency(0.2)).toBe("$0.2");
    expect(formatTickCurrency(0.25)).toBe("$0.25");
    expect(formatTickCurrency(0.999)).toBe("$0.999");
  });

  it("leaves 1-999 without an SI suffix", () => {
    expect(formatTickCurrency(1)).toBe("$1");
    expect(formatTickCurrency(250)).toBe("$250");
    expect(formatTickCurrency(999)).toBe("$999");
  });

  it("uses lowercase k for thousands", () => {
    expect(formatTickCurrency(1000)).toBe("$1k");
    expect(formatTickCurrency(1500)).toBe("$1.5k");
    expect(formatTickCurrency(12345)).toBe("$12.3k");
  });

  it("uses uppercase M for millions", () => {
    expect(formatTickCurrency(1e6)).toBe("$1M");
    expect(formatTickCurrency(2.5e6)).toBe("$2.5M");
    expect(formatTickCurrency(1.23e8)).toBe("$123M");
  });

  it("uses B and T for billions and trillions", () => {
    expect(formatTickCurrency(1e9)).toBe("$1B");
    expect(formatTickCurrency(2.5e9)).toBe("$2.5B");
    expect(formatTickCurrency(1e12)).toBe("$1T");
  });

  it("preserves the sign on negatives", () => {
    expect(formatTickCurrency(-0.1)).toBe("-$0.1");
    expect(formatTickCurrency(-1500)).toBe("-$1.5k");
    expect(formatTickCurrency(-1e6)).toBe("-$1M");
  });

  it("trims trailing zeros from round values", () => {
    expect(formatTickCurrency(100)).toBe("$100");
    expect(formatTickCurrency(1e8)).toBe("$100M");
  });

  it("returns the empty string for non-finite inputs", () => {
    expect(formatTickCurrency(Number.NaN)).toBe("");
    expect(formatTickCurrency(Number.POSITIVE_INFINITY)).toBe("");
    expect(formatTickCurrency(Number.NEGATIVE_INFINITY)).toBe("");
  });
});
