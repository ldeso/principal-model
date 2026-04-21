// Fetch kVCM spot history (Alchemy Prices API) into report/data/.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", "report", "data");
const OUT_PATH = resolve(DATA_DIR, "kvcm-historical.json");

const ADDRESS = "0x00fbac94fec8d4089d3fe979f39454f48c71a65d";
const NETWORK = "base-mainnet";
const INTERVAL = "1d";
// Alchemy caps 1d at 365 points / request.
const WINDOW_DAYS = 365;

interface AlchemyPoint {
  value: string;
  timestamp: string;
}

interface AlchemyResponse {
  network?: string;
  address?: string;
  currency?: string;
  data?: AlchemyPoint[];
  error?: { message?: string };
}

interface OutputPoint {
  timestamp: string;
  value: number;
}

interface Output {
  address: string;
  network: string;
  currency: string;
  interval: string;
  fetchedAt: string;
  data: OutputPoint[];
}

function writeOutput(out: Output): void {
  mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(OUT_PATH, JSON.stringify(out, null, 2));
  console.log(`wrote ${OUT_PATH} (${out.data.length} points)`);
}

async function main(): Promise<void> {
  const apiKey = process.env.ALCHEMY_API_KEY;
  const fetchedAt = new Date().toISOString();

  if (!apiKey) {
    console.warn(
      "ALCHEMY_API_KEY not set; writing empty kvcm-historical.json.",
    );
    writeOutput({
      address: ADDRESS,
      network: NETWORK,
      currency: "usd",
      interval: INTERVAL,
      fetchedAt,
      data: [],
    });
    return;
  }

  const now = new Date();
  const start = new Date(now.getTime() - WINDOW_DAYS * 24 * 60 * 60 * 1000);
  const body = {
    network: NETWORK,
    address: ADDRESS,
    startTime: start.toISOString(),
    endTime: now.toISOString(),
    interval: INTERVAL,
  };

  const url = `https://api.g.alchemy.com/prices/v1/${apiKey}/tokens/historical`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json: AlchemyResponse;
  try {
    json = JSON.parse(text) as AlchemyResponse;
  } catch {
    throw new Error(`Alchemy ${res.status}: non-JSON response: ${text.slice(0, 200)}`);
  }
  if (!res.ok || json.error) {
    throw new Error(
      `Alchemy ${res.status}: ${json.error?.message ?? text.slice(0, 200)}`,
    );
  }

  const data: OutputPoint[] = (json.data ?? []).map((p) => ({
    timestamp: p.timestamp,
    value: Number(p.value),
  }));

  writeOutput({
    address: ADDRESS,
    network: NETWORK,
    currency: json.currency ?? "usd",
    interval: INTERVAL,
    fetchedAt,
    data,
  });
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
