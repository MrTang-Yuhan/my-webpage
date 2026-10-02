export const STATS_KEY = "site-stats-v2";
export const TIME_ZONE = "Asia/Shanghai";

export function dayKey(date = new Date()) {
  return new Date(date.getTime() + 8 * 60 * 60 * 1000).toISOString().slice(0, 10);
}

export function parseDateKey(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value || ''))) return null;
  const [year, month, day] = String(value).split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return String(value);
}

export function emptyStats(legacy = null) {
  return {
    version: 2,
    all: { total: 0, days: {}, pages: {} },
    unique: { total: 0, days: {} },
    missingIpViews: 0,
    startedAt: null,
    updatedAt: null,
    legacy
  };
}

export async function readStats(env) {
  if (!env.SITE_STATS) return emptyStats();
  const raw = await env.SITE_STATS.get(STATS_KEY);
  if (raw) {
    const value = JSON.parse(raw);
    // Never overwrite damaged data with a zero counter.
    if (value?.version !== 2 || !value.all?.days || !value.all?.pages ||
        !value.unique?.days ||
        !Number.isFinite(value.all.total) || !Number.isFinite(value.unique.total)) {
      throw new Error("Invalid statistics");
    }
    return value;
  }
  const oldRaw = await env.SITE_STATS.get("site-stats-v1");
  if (!oldRaw) return emptyStats();
  const old = JSON.parse(oldRaw);
  // Old daily-deduplicated data cannot reconstruct PV or all-time distinct IPs.
  // Preserve it at its original key and report it separately, never as new data.
  return emptyStats({ total: Number(old.total || 0), updatedAt: old.updatedAt || null });
}

export function selectedDay(stats, date, now = new Date()) {
  const today = dayKey(now);
  const requested = parseDateKey(date) || today;
  const startedValue = stats.startedAt ? new Date(stats.startedAt) : null;
  const started = startedValue && !Number.isNaN(startedValue.getTime()) ? dayKey(startedValue) : null;
  let status = 'ok';
  if (requested > today) status = 'future';
  else if (started && requested < started) status = 'before-start';
  else if (!Object.prototype.hasOwnProperty.call(stats.all.days, requested) &&
      !Object.prototype.hasOwnProperty.call(stats.unique.days, requested)) status = 'no-data';
  return {
    date: requested,
    status,
    all: Number(stats.all.days[requested] || 0),
    unique: Number(stats.unique.days[requested] || 0)
  };
}

export function snapshot(stats, now = new Date(), requestedDate = null) {
  const today = dayKey(now);
  const daily = Array.from({ length: 14 }, (_, index) => {
    const date = dayKey(new Date(now.getTime() - (13 - index) * 86400000));
    return { date, count: Number(stats.all.days[date] || 0), unique: Number(stats.unique.days[date] || 0) };
  });
  return {
    version: 2,
    timeZone: TIME_ZONE,
    today,
    allTotal: stats.all.total,
    allToday: Number(stats.all.days[today] || 0),
    uniqueTotal: stats.unique.total,
    uniqueToday: Number(stats.unique.days[today] || 0),
    daily,
    pageCount: Object.keys(stats.all.pages).length,
    pageViews: stats.all.pages,
    missingIpViews: stats.missingIpViews || 0,
    startedAt: stats.startedAt,
    updatedAt: stats.updatedAt,
    legacy: stats.legacy,
    selected: selectedDay(stats, requestedDate || today, now)
  };
}

export function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" }
  });
}
