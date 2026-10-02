import { readStats, snapshot, parseDateKey, json } from "../_lib/stats.js";

export async function onRequestGet({ env, request }) {
  try {
    const stats = await readStats(env);
    const requestedDate = request ? new URL(request.url).searchParams.get('date') : null;
    if (requestedDate && !parseDateKey(requestedDate)) return json({ error: 'Invalid date' }, 400);
    return json({ persistent: Boolean(env.SITE_STATS), ...snapshot(stats, new Date(), requestedDate) });
  } catch (_) {
    return json({ error: "Statistics temporarily unavailable" }, 503);
  }
}
