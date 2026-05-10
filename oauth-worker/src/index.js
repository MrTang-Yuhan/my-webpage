export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }

    if (url.pathname === "/auth") {
      return handleAuth(url, env);
    }

    if (url.pathname === "/callback") {
      return handleCallback(url, env);
    }

    return json({ error: "Not found" }, 404);
  },
};

function handleAuth(url, env) {
  const provider = url.searchParams.get("provider");
  const origin = url.searchParams.get("origin");

  if (provider !== "github") {
    return json({ error: "Only github provider is supported" }, 400);
  }
  if (!origin) {
    return json({ error: "Missing origin" }, 400);
  }

  const state = makeState({ origin });
  const callbackUrl = new URL("/callback", env.AUTH_BASE_URL).toString();
  const githubAuthUrl = new URL("https://github.com/login/oauth/authorize");
  githubAuthUrl.searchParams.set("client_id", env.GITHUB_CLIENT_ID);
  githubAuthUrl.searchParams.set("redirect_uri", callbackUrl);
  githubAuthUrl.searchParams.set("scope", "repo");
  githubAuthUrl.searchParams.set("state", state);
  return Response.redirect(githubAuthUrl.toString(), 302);
}

async function handleCallback(url, env) {
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");

  if (!code || !state) {
    return html(errorPage("Missing code/state query params"));
  }

  const parsedState = parseState(state);
  if (!parsedState) {
    return html(errorPage("Invalid OAuth state"));
  }

  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "User-Agent": "decap-cf-worker",
    },
    body: JSON.stringify({
      client_id: env.GITHUB_CLIENT_ID,
      client_secret: env.GITHUB_CLIENT_SECRET,
      code,
      redirect_uri: new URL("/callback", env.AUTH_BASE_URL).toString(),
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

  const script = `
    (function () {
      function receiveMessage(e) {
        if (e.origin !== ${JSON.stringify(parsedState.origin)}) return;
        const message = 'authorization:github:success:' + JSON.stringify({
          token: tokenJson.access_token,
          provider: "github"
        });
        e.source.postMessage(message, e.origin);
        window.removeEventListener("message", receiveMessage, false);
      }
      window.addEventListener("message", receiveMessage, false);
      window.opener.postMessage("authorizing:github", ${JSON.stringify(parsedState.origin)});
    })();
  `;

  return html(`<!doctype html><html><body><script>${script}</script></body></html>`);
}

function makeState(payload) {
  const jsonPayload = JSON.stringify(payload);
  const payloadB64 = toBase64Url(jsonPayload);
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  const nonce = toBase64Url(String.fromCharCode(...bytes));
  return `${payloadB64}.${nonce}`;
}

function parseState(state) {
  const [payloadB64] = state.split(".");
  if (!payloadB64) return null;
  try {
    const json = fromBase64Url(payloadB64);
    return JSON.parse(json);
  } catch {
    return null;
  }
}

function toBase64Url(input) {
  return btoa(input).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(input) {
  const padded = input.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((input.length + 3) % 4);
  return atob(padded);
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() },
  });
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
