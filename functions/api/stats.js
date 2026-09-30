import { readStats, snapshot, json } from "../_lib/stats.js";

export async function onRequestGet({ env }) {
  try {
    const stats = await readStats(env);
    return json({ persistent: Boolean(env.SITE_STATS), ...snapshot(stats) });
  } catch (_) {
    return json({ error: "Statistics temporarily unavailable" }, 503);
  }
}
