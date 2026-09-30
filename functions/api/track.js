import { STATS_KEY, dayKey, readStats, json } from "../_lib/stats.js";

export async function onRequestPost({ request, env }) {
  const origin = request.headers.get("Origin");
  if (origin && origin !== new URL(request.url).origin) return json({ error: "Invalid origin" }, 403);
  let body;
  try { body = await request.json(); } catch (_) { return json({ error: "Invalid payload" }, 400); }
  const path = normalizePath(body?.path);
  if (!path) return json({ error: "Invalid page path" }, 400);
  if (/^\/(?:admin|stats|api)(?:\/|$)/.test(path)) return json({ ok: true, counted: false });
  if (!env.SITE_STATS) return json({ ok: true, counted: false, persistent: false });

  try {
    const stats = await readStats(env);
    const now = new Date();
    const today = dayKey(now);
    const hash = await visitorHash(request, env);
    let uniqueTotalCounted = false;
    let uniqueTodayCounted = false;
    // Every page load counts, including repeat visits by the same IP.
    stats.all.total += 1;
    stats.all.days[today] = Number(stats.all.days[today] || 0) + 1;
    stats.all.pages[path] = Number(stats.all.pages[path] || 0) + 1;
    if (hash) {
      const totalKey = "site-stats-visitor-total:" + hash;
      const dailyKey = "site-stats-visitor-day:" + today + ":" + hash;
      uniqueTotalCounted = !(await env.SITE_STATS.get(totalKey));
      uniqueTodayCounted = !(await env.SITE_STATS.get(dailyKey));
      if (uniqueTotalCounted) stats.unique.total += 1;
      if (uniqueTodayCounted) stats.unique.days[today] = Number(stats.unique.days[today] || 0) + 1;
      if (uniqueTotalCounted) await env.SITE_STATS.put(totalKey, "1");
      if (uniqueTodayCounted) await env.SITE_STATS.put(dailyKey, "1", { expirationTtl: 172800 });
    } else {
      // Missing IPs contribute PV, but must never be invented as unique visitors.
      stats.missingIpViews += 1;
    }
    stats.startedAt = stats.startedAt || now.toISOString();
    stats.updatedAt = now.toISOString();
    await env.SITE_STATS.put(STATS_KEY, JSON.stringify(stats));
    return json({ ok: true, allCounted: true, uniqueTotalCounted, uniqueTodayCounted, persistent: true });
  } catch (_) {
    return json({ error: "Statistics could not be saved" }, 503);
  }
}

async function visitorHash(request, env) {
  // Trust Cloudflare's header, not a client-supplied forwarding list.
  const ip = (request.headers.get("CF-Connecting-IP") || "").trim().toLowerCase();
  if (!ip) return null;
  const salt = String(env.SITE_STATS_SALT || env.ADMIN_SESSION_SECRET || "site-stats-v2");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(salt + ":" + ip));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

function normalizePath(value) {
  if (typeof value !== "string" || !value.startsWith("/") || value.startsWith("//") ||
      /[\\\u0000-\u0020]/.test(value) || value.length > 4096) return null;
  const path = value.split(/[?#]/)[0];
  try {
    // Do not truncate encoded Chinese URLs or merge encoded slashes into paths.
    return path.split("/").map(part => encodeURIComponent(decodeURIComponent(part)))
      .join("/").replace(/\/+$/, "") || "/";
  } catch (_) { return null; }
}
