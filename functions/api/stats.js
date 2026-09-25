const STATS_KEY = "site-stats-v1";

export async function onRequestGet(context) {
  const { env } = context;
  const stats = await readStats(env);
  const today = new Date().toISOString().slice(0, 10);
  const daily = [];
  for (let offset = 13; offset >= 0; offset -= 1) {
    const date = new Date();
    date.setUTCDate(date.getUTCDate() - offset);
    const key = date.toISOString().slice(0, 10);
    daily.push({ date: key, count: Number(stats.days[key] || 0) });
  }
  const topPages = Object.entries(stats.pages)
    .map(([path, count]) => ({ path, count: Number(count || 0), title: path === "/" ? "首页" : path }))
    .sort((a, b) => b.count - a.count || a.path.localeCompare(b.path))
    .slice(0, 10);
  return json({ persistent: Boolean(env.SITE_STATS), total: stats.total, today: Number(stats.days[today] || 0), daily, topPages, updatedAt: stats.updatedAt });
}

async function readStats(env) {
  if (!env.SITE_STATS) return { total: 0, days: {}, pages: {}, updatedAt: null };
  try {
    const raw = await env.SITE_STATS.get(STATS_KEY);
    const value = raw ? JSON.parse(raw) : {};
    return { total: Number(value.total || 0), days: value.days || {}, pages: value.pages || {}, updatedAt: value.updatedAt || null };
  } catch (_) { return { total: 0, days: {}, pages: {}, updatedAt: null }; }
}
function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
}
