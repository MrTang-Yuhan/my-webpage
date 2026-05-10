export async function onRequestGet(context) {
  const { request, env } = context;
  const url = new URL(request.url);

  const provider = url.searchParams.get("provider");
  const origin = url.searchParams.get("origin");
  if (provider !== "github") {
    return json({ error: "Only github provider is supported" }, 400);
  }
  if (!origin) {
    return json({ error: "Missing origin" }, 400);
  }

  const state = makeState({ origin });
  const callback = "https://my-webpage-adu.pages.dev/api/callback";

  const github = new URL("https://github.com/login/oauth/authorize");
  github.searchParams.set("client_id", env.GITHUB_CLIENT_ID || "");
  github.searchParams.set("redirect_uri", callback);
  github.searchParams.set("scope", "repo");
  github.searchParams.set("state", state);

  return Response.redirect(github.toString(), 302);
}

function makeState(payload) {
  const payloadB64 = toBase64Url(JSON.stringify(payload));
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  const nonce = toBase64Url(String.fromCharCode(...bytes));
  return `${payloadB64}.${nonce}`;
}

function toBase64Url(input) {
  return btoa(input).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
