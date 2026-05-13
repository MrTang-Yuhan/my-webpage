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

  return html(`<!doctype html><html><body><script>${script}</script></body></html>`);
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

function html(content, status = 200) {
  return new Response(content, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}

function errorPage(message) {
  const safe = String(message).replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `<!doctype html><html><body><h1>OAuth Error</h1><pre>${safe}</pre></body></html>`;
}
