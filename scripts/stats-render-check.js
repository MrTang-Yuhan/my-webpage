const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const http = require('node:http');

const root = path.resolve(__dirname, '..');
const site = path.join(root, '_site');
const source = fs.readFileSync(path.join(root, 'src/js/stats.js'), 'utf8');
const html = fs.readFileSync(path.join(site, 'stats/index.html'), 'utf8');
assert.equal(fs.readFileSync(path.join(site, 'js/stats.js'), 'utf8'), source, 'Run npm run build before checking stats');
const template = html.match(/<template id="stats-page-titles">([\s\S]*?)<\/template>/)[1];
function decode(value) {
  return value.replace(/&(#(?:x[\da-f]+|\d+)|amp|quot|lt|gt|apos);/gi, (_, entity) => {
    if (entity[0] === '#') return String.fromCodePoint(entity[1] === 'x' ? parseInt(entity.slice(2), 16) : Number(entity.slice(1)));
    return { amp: '&', quot: '"', lt: '<', gt: '>', apos: "'" }[entity];
  });
}
const links = [...template.matchAll(/<a href="([^"]*)">([\s\S]*?)<\/a>/g)].map(match => ({
  href: decode(match[1]), textContent: decode(match[2]), getAttribute() { return this.href; }
}));
assert.ok(links.every(link => link.href.startsWith('/posts/') && link.href !== '/posts/'), 'Title index must contain only article pages');
assert.ok(html.includes('<h2>热门文章</h2>'));
const article = links.find(link => link.href.includes('tensor-parallelism张量并行（一）'));
assert.ok(article, 'The screenshot article must be present in the title index');
const encoded = encodeURI(article.href);
const today = new Date().toISOString().slice(0, 10);
const fixture = {
  persistent: true, total: 6, today: 1, pageCount: 2,
  daily: Array.from({ length: 14 }, (_, index) => {
    const date = new Date();
    date.setUTCDate(date.getUTCDate() - 13 + index);
    return { date: date.toISOString().slice(0, 10), count: index < 8 ? 0 : 1 };
  }),
  topPages: [{ path: '/', count: 5, title: '首页' }, { path: encoded, title: encoded, count: 1 }],
  updatedAt: new Date().toISOString()
};
const stress = {
  ...fixture, total: 987654321, pageCount: 25,
  daily: fixture.daily.map((item, i) => ({ ...item, count: i * 123456 })),
  topPages: [
    ...fixture.topPages,
    ...links.slice(0, 8).map((link, i) => ({ path: encodeURI(link.href), count: 123456789 - i })),
    { path: '/unknown/' + '超长中文标题'.repeat(60) + '/', count: 123456789 },
    { path: '/unknown/' + 'UnbrokenEnglishTitle'.repeat(30) + '/', count: 12345 },
    { path: '/unknown/' + encodeURIComponent('没有索引的中文文章') + '/', count: 4 },
    { path: '/broken/%E0%A4%A/', count: 3 },
    { path: '/quoted/%22%3Cscript%3E/', title: '<script>alert("test")</script>', count: 2 }
  ]
};

async function render(data, local = '{}', index = links) {
  const nodes = Object.fromEntries(['total', 'today', 'pages', 'daily', 'topPages', 'updated'].map(key => [key, { textContent: '', innerHTML: '' }]));
  let rejectError;
  const done = new Promise(resolve => {
    Object.defineProperty(nodes.updated, 'textContent', { set(value) { this.value = value; resolve(); } });
  });
  vm.runInNewContext(source, {
    document: {
      querySelector: () => ({ querySelector: selector => nodes[selector.match(/"([^"]+)"/)[1]] }),
      getElementById: () => ({ content: { querySelectorAll: () => index } })
    },
    localStorage: { getItem: () => local },
    fetch: async () => {
      if (data instanceof Error) throw data;
      return { ok: true, json: async () => JSON.parse(JSON.stringify(data)) };
    },
    Intl, Date, console
  });
  let timeout;
  await Promise.race([done, new Promise((_, reject) => { rejectError = reject; timeout = setTimeout(() => rejectError(new Error('Stats render did not complete')), 2000); })])
    .finally(() => clearTimeout(timeout));
  return nodes;
}

async function check() {
  const serverSource = fs.readFileSync(path.join(root, 'functions/api/stats.js'), 'utf8');
  const { onRequestGet } = await import('data:text/javascript;base64,' + Buffer.from(serverSource).toString('base64'));
  const pages = Object.fromEntries(links.slice(0, 15).map((link, i) => [link.href, i + 1]));
  const generalPages = Object.fromEntries(['/', '/posts/', '/about/', '/stats/', '/admin/', '/posts/missing/deleted/', article.href + 'attach/demo.html',
    ...Array.from({ length: 12 }, (_, i) => '/landing-' + i + '/')].map(url => [url, 9999]));
  Object.assign(pages, generalPages);
  const response = await onRequestGet({ env: { SITE_STATS: { get: async () => JSON.stringify({ total: 120, pages, days: { [today]: 7 } }) } } });
  const api = await response.json();
  assert.equal(api.pageCount, Object.keys(pages).length);
  assert.equal(api.topPages.length, 10);
  assert.deepEqual(api.pageViews, pages);
  const serverNodes = await render(api);
  assert.equal(serverNodes.today.textContent, '7');
  assert.equal(serverNodes.pages.textContent, String(Object.keys(pages).length));
  assert.equal((serverNodes.topPages.innerHTML.match(/<li>/g) || []).length, 10, 'General pages must not crowd out articles');
  assert.ok(decode(serverNodes.topPages.innerHTML).includes(links[14].textContent.trim()), 'Highest-view article should lead');
  for (const url of Object.keys(generalPages)) {
    assert.ok(!decode(serverNodes.topPages.innerHTML).includes('href="' + url + '"'), 'Non-article in ranking: ' + url);
  }
  assert.match(serverNodes.daily.innerHTML, /stats-bar-value">7</);
  for (const link of links) {
    for (const variant of [link.href, encodeURI(link.href), encodeURI(link.href).replace(/%[A-F0-9]{2}/g, s => s.toLowerCase()).replace(/\/$/, '') || '/']) {
      const nodes = await render({ ...fixture, topPages: [{ path: variant, title: variant, count: 1 }] });
      const title = decode(nodes.topPages.innerHTML.match(/class="stats-ranking-title">([\s\S]*?)<\/span>/)[1]);
      assert.equal(title, link.textContent.trim(), 'Title lookup failed for ' + variant);
      assert.equal(decode(nodes.topPages.innerHTML.match(/href="([^"]+)"/)[1]), variant, 'Navigation URL must remain unchanged');
    }
  }
  const local = JSON.stringify({ total: 120, days: { [today]: 8 }, pages: { ...pages, [encoded]: 100 } });
  for (const data of [new Error('offline'), { persistent: false }]) {
    const nodes = await render(data, local);
    assert.equal(nodes.today.textContent, '8');
    assert.equal(nodes.pages.textContent, String(Object.keys({ ...pages, [encoded]: 100 }).length));
    assert.equal((nodes.topPages.innerHTML.match(/<li>/g) || []).length, 10);
    assert.ok(decode(nodes.topPages.innerHTML).includes(article.textContent));
  }
  const empty = await render(new Error('offline'), '{broken');
  assert.equal(empty.total.textContent, '0');
  assert.ok(empty.topPages.innerHTML.includes('暂无文章访问记录'));
  const stressNodes = await render(stress);
  assert.ok(!stressNodes.topPages.innerHTML.includes('没有索引的中文文章'));
  assert.ok(!stressNodes.topPages.innerHTML.includes('<script>'));
  const unsafe = await render({ ...fixture, topPages: [{ path: 'javascript:alert(1)', count: 1 }, { path: '//example.com', count: 1 }] });
  assert.ok(!unsafe.topPages.innerHTML.includes('href='));
  const noIndex = await render(fixture, '{}', []);
  assert.ok(noIndex.topPages.innerHTML.includes('暂无文章访问记录'));
  const onlyGeneral = await render({ ...fixture, pageViews: generalPages });
  assert.ok(onlyGeneral.topPages.innerHTML.includes('暂无文章访问记录'));
  const onlyGeneralLocal = await render(new Error('offline'), JSON.stringify({ pages: generalPages }));
  assert.ok(onlyGeneralLocal.topPages.innerHTML.includes('暂无文章访问记录'));
  const merged = await render({ ...fixture, pageViews: { [article.href]: 3, [encoded]: 4, '/': 999 } });
  assert.equal((merged.topPages.innerHTML.match(/<li>/g) || []).length, 1);
  assert.match(merged.topPages.innerHTML, /stats-ranking-count">7</);
  const special = { href: '/posts/testing/special/', textContent: '<script>alert("test")</script>', getAttribute() { return this.href; } };
  const escaped = await render({ ...fixture, pageViews: { [special.href]: 1 } }, '{}', [special]);
  assert.ok(escaped.topPages.innerHTML.includes('&lt;script&gt;'));
  assert.ok(!escaped.topPages.innerHTML.includes('<script>'));
  console.log('Stats checks passed: ' + links.length + ' articles × 3 URL forms; article-only ranking before top-ten limit, merged paths, API/local/offline, totals and HTML escaping.');
}

const geometry = function () {
  const errors = [];
  const tolerance = 1;
  const box = element => element.getBoundingClientRect();
  if (document.documentElement.scrollWidth > document.documentElement.clientWidth + tolerance) errors.push('page overflow');
  for (const panel of document.querySelectorAll('.stats-panel')) {
    if (panel.scrollWidth > panel.clientWidth + tolerance) errors.push('panel overflow');
  }
  for (const row of document.querySelectorAll('.stats-ranking li')) {
    const label = row.querySelector('a, .stats-ranking-title');
    const counter = row.querySelector('.stats-ranking-count');
    if (counter && label && box(label).right > box(counter).left + tolerance) errors.push('title/count overlap');
    const title = row.querySelector('.stats-ranking-title');
    if (title && box(title).height > parseFloat(getComputedStyle(title).lineHeight) * 2 + tolerance) errors.push('title exceeds two lines');
  }
  const labels = [...document.querySelectorAll('.stats-bar-label')].filter(el => getComputedStyle(el).visibility !== 'hidden');
  for (let i = 1; i < labels.length; i++) {
    if (box(labels[i - 1]).right > box(labels[i]).left + tolerance) errors.push('date overlap');
  }
  return errors;
};

function serve() {
  http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost:4173');
    res.setHeader('Cache-Control', 'no-store');
    if (url.pathname === '/api/stats') {
      res.setHeader('Content-Type', 'application/json');
      return res.end(JSON.stringify((req.headers.referer || '').includes('fixture=stress') ? stress : fixture));
    }
    if (url.pathname === '/__stats-check/') {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      return res.end('<!doctype html><meta charset="utf-8"><h1>Stats layout regression</h1><pre id="results">Running…</pre><div id="frames"></div><script>' +
        'const results = []; const check = ' + geometry.toString() + ';' +
        '(async () => { for (const width of [320, 375, 640, 768, 820, 821, 1024, 1440]) { for (const theme of ["light", "dark"]) {' +
        'const frame = document.createElement("iframe"); frame.width = width; frame.height = 1200; frame.style.border = "0";' +
        'const ready = new Promise(resolve => frame.onload = resolve); frame.src = "/stats/?fixture=stress"; document.getElementById("frames").append(frame); await ready;' +
        'const doc = frame.contentDocument; doc.documentElement.dataset.theme = theme; await doc.fonts.ready;' +
        'for (let i=0; i<100 && doc.querySelector("[data-stat=total]").textContent === "加载中"; i++) await new Promise(r => setTimeout(r, 20));' +
        'const errors = frame.contentWindow.eval("(" + check.toString() + ")()"); if(doc.querySelector("[data-stat=total]").textContent === "加载中") errors.push("render timeout");' +
        'results.push({width, theme, errors}); document.getElementById("results").textContent = JSON.stringify(results, null, 2); frame.remove();' +
        '} } document.title = results.some(r=>r.errors.length) ? "FAIL" : "PASS"; })();</script>');
    }
    let filename;
    try { filename = path.resolve(site, '.' + decodeURIComponent(url.pathname)); }
    catch (_) { res.statusCode = 400; return res.end(); }
    if (!filename.startsWith(site + path.sep) && filename !== site) { res.statusCode = 403; return res.end(); }
    if (fs.existsSync(filename) && fs.statSync(filename).isDirectory()) filename = path.join(filename, 'index.html');
    if (!fs.existsSync(filename)) { res.statusCode = 404; return res.end(); }
    const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.png': 'image/png' };
    res.setHeader('Content-Type', types[path.extname(filename)] || 'application/octet-stream');
    fs.createReadStream(filename).pipe(res);
  }).listen(4173, '127.0.0.1', () => console.log('Fixture preview: http://127.0.0.1:4173/stats/ | Layout checks: http://127.0.0.1:4173/__stats-check/'));
}

check().then(() => { if (process.argv.includes('--serve')) serve(); }).catch(error => { console.error(error); process.exitCode = 1; });
