export async function onRequestPost(context) {
  const { request, env } = context;
  try {
    const sameOriginCheck = validateSameOrigin(request);
    if (!sameOriginCheck.ok) {
      return json({ error: sameOriginCheck.error }, 403);
    }

    const sessionSecret = resolveSessionSecret(env);
    if (!sessionSecret) {
      return json({ error: "Server missing session secret." }, 500);
    }
    const session = await verifyAdminSessionFromRequest(request, sessionSecret);
    if (!session.ok) {
      return json({ error: session.error }, 401);
    }

    const body = await request.json();
    const token = resolveGithubToken(request, env);
    if (!token) {
      return json({ error: "Missing write token. Set GITHUB_ADMIN_TOKEN on server." }, 500);
    }

    const repo = String(body.repo || "").trim();
    const branch = String(body.branch || "main").trim();
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) {
      return json({ error: "Invalid repo. Expected owner/repo." }, 400);
    }
    if (!/^[A-Za-z0-9._/-]+$/.test(branch)) {
      return json({ error: "Invalid branch name." }, 400);
    }

    const targetArchive = sanitizeArchiveName(body.targetArchive);
    if (!targetArchive) {
      return json({ error: "Invalid targetArchive." }, 400);
    }

    const slug = sanitizeSlug(body.slug);
    const oldArchive = sanitizeArchiveName(body.oldArchive);
    if (!slug || !oldArchive) {
      return json({ error: "Missing or invalid oldArchive/slug." }, 400);
    }

    const oldDir = `src/posts/${oldArchive}/${slug}`;
    const newDir = `src/posts/${targetArchive}/${slug}`;
    if (oldDir === newDir) {
      return json({ error: "Source and target directories are identical." }, 400);
    }

    const ref = await githubApi(`/repos/${repo}/git/ref/heads/${encodeURIComponent(branch)}`, { method: "GET" }, token);
    const baseCommitSha = String(ref?.object?.sha || "");
    if (!baseCommitSha) return json({ error: "Failed to resolve branch head commit." }, 500);

    const baseCommit = await githubApi(`/repos/${repo}/git/commits/${baseCommitSha}`, { method: "GET" }, token);
    const baseTreeSha = String(baseCommit?.tree?.sha || "");
    if (!baseTreeSha) return json({ error: "Failed to resolve base tree." }, 500);

    const tree = await githubApi(
      `/repos/${repo}/git/trees/${baseTreeSha}?recursive=1`,
      { method: "GET" },
      token
    );
    const nodes = Array.isArray(tree?.tree) ? tree.tree : [];

    const oldPrefix = `${oldDir}/`;
    const newPrefix = `${newDir}/`;
    const oldFiles = nodes.filter((node) => node.type === "blob" && String(node.path || "").startsWith(oldPrefix));
    if (!oldFiles.length) return json({ error: `Source directory not found: ${oldDir}` }, 404);
    if (!oldFiles.some((n) => String(n.path || "").toLowerCase() === `${oldDir}/index.md`.toLowerCase())) {
      return json({ error: `Missing index.md in source directory: ${oldDir}` }, 400);
    }
    if (nodes.some((node) => String(node.path || "").startsWith(newPrefix))) {
      return json({ error: `Target directory already exists: ${newDir}` }, 409);
    }

    const patchEntries = [];
    const changedIndexPath = `${oldDir}/index.md`;
    for (const file of oldFiles) {
      const oldPath = String(file.path || "");
      const suffix = oldPath.slice(oldPrefix.length);
      const newPath = `${newPrefix}${suffix}`;
      const mode = String(file.mode || "100644");
      const sha = String(file.sha || "");
      if (!sha) return json({ error: `Missing blob sha for ${oldPath}` }, 500);

      if (oldPath === changedIndexPath) {
        const oldBlob = await githubApi(`/repos/${repo}/git/blobs/${encodeURIComponent(sha)}`, { method: "GET" }, token);
        const oldContentB64 = normalizeBase64(String(oldBlob?.content || ""));
        if (String(oldBlob?.encoding || "").toLowerCase() !== "base64" || !oldContentB64) {
          return json({ error: `Failed to read index.md blob content: ${oldPath}` }, 500);
        }
        const movedMarkdown = updateFrontMatterArchive(base64ToUtf8(oldContentB64), targetArchive);
        const newBlob = await githubApi(
          `/repos/${repo}/git/blobs`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: utf8ToBase64(movedMarkdown), encoding: "base64" }),
          },
          token
        );
        patchEntries.push({ path: newPath, mode, type: "blob", sha: String(newBlob.sha || "") });
      } else {
        patchEntries.push({ path: newPath, mode, type: "blob", sha });
      }

      patchEntries.push({ path: oldPath, mode, type: "blob", sha: null });
    }

    const newTree = await githubApi(
      `/repos/${repo}/git/trees`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_tree: baseTreeSha, tree: patchEntries }),
      },
      token
    );
    const newTreeSha = String(newTree?.sha || "");
    if (!newTreeSha) return json({ error: "Failed to create new git tree." }, 500);

    const commitMessage = `chore(admin): move post archive ${slug} to ${targetArchive}`;
    const newCommit = await githubApi(
      `/repos/${repo}/git/commits`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: commitMessage,
          tree: newTreeSha,
          parents: [baseCommitSha],
        }),
      },
      token
    );
    const newCommitSha = String(newCommit?.sha || "");
    if (!newCommitSha) return json({ error: "Failed to create commit." }, 500);

    await githubApi(
      `/repos/${repo}/git/refs/heads/${encodeURIComponent(branch)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sha: newCommitSha, force: false }),
      },
      token
    );

    return json({
      ok: true,
      movedCount: oldFiles.length,
      oldDir,
      newDir,
      commitSha: newCommitSha,
      commitUrl: `https://github.com/${repo}/commit/${newCommitSha}`,
    });
  } catch (err) {
    return json({ error: String(err && err.message ? err.message : err || "unknown error") }, 500);
  }
}

export async function onRequestOptions() {
  return new Response(null, { status: 204, headers: corsHeaders() });
}

function resolveSessionSecret(env) {
  const primary = String(env.ADMIN_SESSION_SECRET || "").trim();
  if (primary) return primary;
  const fallback = String(env.GITHUB_CLIENT_SECRET || "").trim();
  return fallback || "";
}

function resolveGithubToken(request, env) {
  const envToken = String(env.GITHUB_ADMIN_TOKEN || "").trim();
  if (envToken) return envToken;
  const auth = String(request.headers.get("authorization") || "");
  const m = auth.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : "";
}

function validateSameOrigin(request) {
  const reqUrl = new URL(request.url);
  const expectedOrigin = reqUrl.origin;
  const origin = request.headers.get("origin");
  if (origin && origin !== expectedOrigin) {
    return { ok: false, error: "Cross-origin request blocked (origin mismatch)." };
  }
  const referer = request.headers.get("referer");
  if (referer) {
    try {
      const refererOrigin = new URL(referer).origin;
      if (refererOrigin !== expectedOrigin) {
        return { ok: false, error: "Cross-origin request blocked (referer mismatch)." };
      }
    } catch {
      return { ok: false, error: "Invalid referer header." };
    }
  }
  return { ok: true };
}

async function verifyAdminSessionFromRequest(request, secret) {
  const cookieHeader = String(request.headers.get("cookie") || "");
  const sessionValue = getCookieValue(cookieHeader, "admin_session");
  if (!sessionValue) return { ok: false, error: "Not logged in. Missing admin session." };

  const parts = sessionValue.split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return { ok: false, error: "Invalid admin session format." };
  }
  const [payloadB64, signature] = parts;
  const expectedSig = await signHmacSha256(payloadB64, secret);
  if (!timingSafeEqual(signature, expectedSig)) {
    return { ok: false, error: "Invalid admin session signature." };
  }

  let payload;
  try {
    payload = JSON.parse(fromBase64Url(payloadB64));
  } catch {
    return { ok: false, error: "Invalid admin session payload." };
  }
  const exp = Number(payload && payload.exp);
  if (!Number.isFinite(exp) || exp <= 0) {
    return { ok: false, error: "Invalid admin session expiration." };
  }
  if (Math.floor(Date.now() / 1000) >= exp) {
    return { ok: false, error: "Admin session expired. Please login again." };
  }
  if (!payload.login || !payload.id) {
    return { ok: false, error: "Invalid admin session user." };
  }
  return { ok: true, payload };
}

function getCookieValue(cookieHeader, name) {
  const chunks = String(cookieHeader || "").split(";");
  for (const chunk of chunks) {
    const [k, ...rest] = chunk.trim().split("=");
    if (k !== name) continue;
    return rest.join("=");
  }
  return "";
}

async function githubApi(path, options, token) {
  const res = await fetch(`https://api.github.com${path}`, {
    ...(options || {}),
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "pages-function-admin-move-post",
      ...((options && options.headers) || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GitHub API ${res.status}: ${text}`);
  }
  return res.status === 204 ? null : res.json();
}

function sanitizeArchiveName(value) {
  const name = String(value || "").trim();
  if (!name) return "";
  if (name.length > 80) return "";
  if (!/^[\p{L}\p{N}_-]+$/u.test(name)) return "";
  return name;
}

function sanitizeSlug(value) {
  const slug = String(value || "").trim();
  if (!slug) return "";
  if (slug.includes("/") || slug.includes("\\")) return "";
  if (slug.length > 200) return "";
  return slug;
}

function normalizeBase64(value) {
  return String(value || "").replace(/\s+/g, "");
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

function fromBase64Url(input) {
  const src = String(input || "");
  const padded = src.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((src.length + 3) % 4);
  return atob(padded);
}

function bytesToBase64Url(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function timingSafeEqual(a, b) {
  const x = String(a || "");
  const y = String(b || "");
  if (x.length !== y.length) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i += 1) {
    diff |= x.charCodeAt(i) ^ y.charCodeAt(i);
  }
  return diff === 0;
}

function base64ToUtf8(base64) {
  const binary = atob(normalizeBase64(base64));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new TextDecoder("utf-8").decode(bytes);
}

function utf8ToBase64(text) {
  const bytes = new TextEncoder().encode(String(text || ""));
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function updateFrontMatterArchive(markdown, archive) {
  const text = String(markdown || "");
  const targetLine = `archive: ${archive}`;
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return `---\n${targetLine}\n---\n\n${text}`;
  const body = match[1];
  const nextBody = /^archive\s*:/m.test(body)
    ? body.replace(/^archive\s*:.*$/m, targetLine)
    : `${body}\n${targetLine}`;
  return text.replace(/^---\r?\n[\s\S]*?\r?\n---/, `---\n${nextBody}\n---`);
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...corsHeaders() },
  });
}
