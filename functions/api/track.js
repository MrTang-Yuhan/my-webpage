const STATS_KEY = "site-stats-v1";

export async function onRequestPost(context) {
  const { request, env } = context;
  if (!isSameOrigin(request)) return json({ error: "Invalid origin" }, 403);

  let body = {};
  try { body = await request.json(); } catch (_) {}
  const path = normalizePath(body.path || "/");
  const today = new Date().toISOString().slice(0, 10);
  const stats = await readStats(env);
  const visitorKey = await getVisitorKey(request, env, today);
  if (visitorKey && env.SITE_STATS) {
    const alreadyCounted = await env.SITE_STATS.get(visitorKey);
    if (alreadyCounted) return json({ ok: true, counted: false, persistent: true });
    await env.SITE_STATS.put(visitorKey, "1", { expirationTtl: 172800 });
  }
  stats.total += 1;
  stats.days[today] = Number(stats.days[today] || 0) + 1;
  stats.pages[path] = Number(stats.pages[path] || 0) + 1;
  stats.updatedAt = new Date().toISOString();
  try { await writeStats(env, stats); } catch (_) {}
  return json({ ok: true, counted: true, persistent: Boolean(env.SITE_STATS) });
}

async function getVisitorKey(request, env, date) {
  const forwarded = request.headers.get("CF-Connecting-IP") || request.headers.get("X-Forwarded-For") || "";
  const ip = String(forwarded).split(",")[0].trim();
  if (!ip) return "";
  const salt = String(env.SITE_STATS_SALT || env.ADMIN_SESSION_SECRET || "site-stats-v1");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${salt}:${date}:${ip}`));
  const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `site-stats-visitor:${date}:${hash}`;
}

function normalizePath(value) {
  const path = String(value || "/").trim();
  if (!path.startsWith("/") || path.startsWith("//") || path.includes("\0")) return "/";
  return path.slice(0, 300);
}

function isSameOrigin(request) {
  const origin = request.headers.get("Origin");
  return !origin || origin === new URL(request.url).origin;
}

async function readStats(env) {
  if (!env.SITE_STATS) return emptyStats();
  try {
    const raw = await env.SITE_STATS.get(STATS_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && typeof parsed === "object") return normalizeStats(parsed);
  } catch (_) {}
  return emptyStats();
}

async function writeStats(env, stats) {
  if (!env.SITE_STATS) return;
  await env.SITE_STATS.put(STATS_KEY, JSON.stringify(stats));
}

function emptyStats() { return { total: 0, days: {}, pages: {}, updatedAt: null }; }
function normalizeStats(value) {
  return {
    total: Number(value.total || 0),
    days: value.days && typeof value.days === "object" ? value.days : {},
    pages: value.pages && typeof value.pages === "object" ? value.pages : {},
    updatedAt: value.updatedAt || null
  };
}
function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
}
