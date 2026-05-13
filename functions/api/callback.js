export async function onRequestGet(context) {
  const { request, env } = context;
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");

  if (!code || !state) {
    return html(errorPage("Missing code/state query params"));
  }

  const parsedState = parseState(state);
  if (!parsedState || !parsedState.origin) {
    return html(errorPage("Invalid OAuth state"));
  }

  const callbackUrl = new URL("/api/callback", resolveAuthBaseUrl(request, env)).toString();

  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "User-Agent": "decap-pages-function",
    },
    body: JSON.stringify({
      client_id: env.GITHUB_CLIENT_ID,
      client_secret: env.GITHUB_CLIENT_SECRET,
      code,
      redirect_uri: callbackUrl,
      state,
    }),
  });

  if (!tokenRes.ok) {
    return html(errorPage(`GitHub token exchange failed: ${tokenRes.status}`));
  }

  const tokenJson = await tokenRes.json();
  if (!tokenJson.access_token) {
    return html(errorPage(`No access_token returned: ${JSON.stringify(tokenJson)}`));
  }

  const userRes = await fetch("https://api.github.com/user", {
    method: "GET",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${tokenJson.access_token}`,
      "User-Agent": "decap-pages-function",
    },
  });
  if (!userRes.ok) {
    return html(errorPage(`GitHub user lookup failed: ${userRes.status}`));
  }
  const userJson = await userRes.json();
  const login = String(userJson?.login || "").trim();
  const id = Number(userJson?.id || 0);
  if (!login || !Number.isFinite(id) || id <= 0) {
    return html(errorPage("GitHub user lookup missing login/id"));
  }

  const secret = resolveSessionSecret(env);
  if (!secret) {
    return html(errorPage("Missing ADMIN_SESSION_SECRET or GITHUB_CLIENT_SECRET for admin session"));
  }
  const cookie = await buildAdminSessionCookie({
    login,
    id,
    ttlSeconds: 60 * 60 * 12,
    secret,
  });

  const origin = parsedState.origin;
  const script = `
    (function () {
      function receiveMessage(e) {
        if (e.origin !== ${JSON.stringify(origin)}) return;
        const message = 'authorization:github:success:' + JSON.stringify({
          token: ${JSON.stringify(tokenJson.access_token)},
          provider: "github"
        });
        e.source.postMessage(message, e.origin);
        window.removeEventListener("message", receiveMessage, false);
      }
      window.addEventListener("message", receiveMessage, false);
      window.opener.postMessage("authorizing:github", ${JSON.stringify(origin)});
    })();
  `;

  return html(`<!doctype html><html><body><script>${script}</script></body></html>`, 200, {
    "Set-Cookie": cookie,
  });
}

function parseState(state) {
  const [payloadB64] = state.split(".");
  if (!payloadB64) return null;
  try {
    return JSON.parse(fromBase64Url(payloadB64));
  } catch {
    return null;
  }
}

function resolveAuthBaseUrl(request, env) {
  const configured = String(env.AUTH_BASE_URL || "").trim();
  if (configured) {
    try {
      return new URL(configured).origin;
    } catch {
      // ignore invalid configured URL
    }
  }
  return new URL(request.url).origin;
}

function fromBase64Url(input) {
  const padded = input.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((input.length + 3) % 4);
  return atob(padded);
}

function html(content, status = 200, extraHeaders = {}) {
  return new Response(content, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8", ...extraHeaders },
  });
}

function errorPage(message) {
  const safe = String(message).replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `<!doctype html><html><body><h1>OAuth Error</h1><pre>${safe}</pre></body></html>`;
}

function resolveSessionSecret(env) {
  const primary = String(env.ADMIN_SESSION_SECRET || "").trim();
  if (primary) return primary;
  const fallback = String(env.GITHUB_CLIENT_SECRET || "").trim();
  return fallback || "";
}

async function buildAdminSessionCookie(args) {
  const payload = {
    login: args.login,
    id: args.id,
    exp: Math.floor(Date.now() / 1000) + Number(args.ttlSeconds || 0),
  };
  const payloadB64 = toBase64Url(JSON.stringify(payload));
  const signature = await signHmacSha256(payloadB64, String(args.secret || ""));
  const cookieValue = `${payloadB64}.${signature}`;
  return `admin_session=${cookieValue}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${Number(
    args.ttlSeconds || 0
  )}`;
}

async function signHmacSha256(message, secret) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(String(message || "")));
  return bytesToBase64Url(new Uint8Array(sig));
}

function bytesToBase64Url(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function toBase64Url(input) {
  return btoa(String(input || "")).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
