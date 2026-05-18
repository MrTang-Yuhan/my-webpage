const fs = require('fs');
const path = require('path');

const root = process.cwd();
const postsRoot = path.join(root, 'src', 'posts');
const adminArchivesPath = path.join(root, '_site', 'admin-archives.json');
const adminConfigPath = path.join(root, 'src', 'admin', 'config.yml');
const adminIndexPath = path.join(root, 'src', 'admin', 'index.html');

function walkIndexFiles(dir, out) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkIndexFiles(full, out);
      continue;
    }
    if (entry.isFile() && entry.name === 'index.md') out.push(full);
  }
}

function readFrontMatterArchive(filePath) {
  const text = fs.readFileSync(filePath, 'utf8');
  const fm = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!fm) return '';
  const archiveLine = fm[1].match(/^archive:\s*(.*)\s*$/m);
  return archiveLine ? archiveLine[1].trim() : '';
}

function normalizeSlash(p) {
  return String(p).replace(/\\/g, '/');
}

function listTopLevelArchiveDirs(indexFiles) {
  const set = new Set();
  for (const file of indexFiles) {
    const rel = normalizeSlash(path.relative(root, file));
    const m = rel.match(/^src\/posts\/([^/]+)\/[^/]+\/index\.md$/);
    if (m) set.add(m[1]);
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));
}

function main() {
  if (!fs.existsSync(postsRoot)) {
    throw new Error('Missing src/posts directory');
  }
  if (!fs.existsSync(adminConfigPath)) {
    throw new Error('Missing src/admin/config.yml');
  }
  if (!fs.existsSync(adminIndexPath)) {
    throw new Error('Missing src/admin/index.html');
  }

  const indexFiles = [];
  walkIndexFiles(postsRoot, indexFiles);

  const mismatches = [];
  for (const file of indexFiles) {
    const rel = normalizeSlash(path.relative(root, file));
    const m = rel.match(/^src\/posts\/([^/]+)\/[^/]+\/index\.md$/);
    if (!m) continue;
    const expectedArchive = m[1];
    const actualArchive = readFrontMatterArchive(file);
    if (actualArchive !== expectedArchive) {
      mismatches.push({ file: rel, expectedArchive, actualArchive });
    }
  }

  if (mismatches.length) {
    console.error('Archive mismatch detected:');
    for (const item of mismatches) {
      console.error(`- ${item.file}: front matter archive="${item.actualArchive}" expected="${item.expectedArchive}"`);
    }
    process.exit(1);
  }

  if (!fs.existsSync(adminArchivesPath)) {
    throw new Error('Missing _site/admin-archives.json. Run build first.');
  }

  const rawJson = fs.readFileSync(adminArchivesPath, 'utf8');
  let parsed;
  try {
    parsed = JSON.parse(rawJson);
  } catch (err) {
    throw new Error('Failed to parse _site/admin-archives.json: ' + err.message);
  }
  if (!Array.isArray(parsed)) {
    throw new Error('_site/admin-archives.json must be an array');
  }
  if (!parsed.every((v) => typeof v === 'string')) {
    throw new Error('_site/admin-archives.json must be string[]');
  }

  const expectedDirs = listTopLevelArchiveDirs(indexFiles);
  const actualDirs = Array.from(new Set(parsed)).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));
  if (JSON.stringify(expectedDirs) !== JSON.stringify(actualDirs)) {
    throw new Error('admin-archives.json directories mismatch. expected=' + JSON.stringify(expectedDirs) + ' actual=' + JSON.stringify(actualDirs));
  }

  const configText = fs.readFileSync(adminConfigPath, 'utf8');
  if (!/name:\s*body[\s\S]*?widget:\s*markdown[\s\S]*?modes:\s*\n\s*-\s*raw\b/.test(configText)) {
    throw new Error('posts body markdown widget must keep raw mode enabled.');
  }
  if (/name:\s*body[\s\S]*?widget:\s*markdown[\s\S]*?modes:\s*[\s\S]*?-\s*rich_text\b/.test(configText)) {
    throw new Error('posts body markdown widget must not expose Rich Text mode.');
  }

  const adminIndexText = fs.readFileSync(adminIndexPath, 'utf8');
  if (/inline-image-button-wrap|inline-image-insert-btn|inline-image-file-input|在线插入|在线插入图片/.test(adminIndexText)) {
    throw new Error('admin editor must not show the retired custom inline image insert module.');
  }
  if (/\/api\/admin\/upload-image/.test(adminIndexText)) {
    throw new Error('admin editor must not call the retired custom upload-image API.');
  }
  if (!/upload-image-only-panel/.test(adminIndexText) || !/findGlobalMediaButton/.test(adminIndexText)) {
    throw new Error('admin editor must expose an upload-only image button that opens Decap native media.');
  }
  if (!/startUploadOnlyMarkdownGuard/.test(adminIndexText)) {
    throw new Error('admin upload-only image button must guard the Markdown body from any edits.');
  }
  if (/protectMarkdownSourceForRichTextImages|restoreMarkdownSourceExceptNewImages|beginRichTextImageSession|richTextActive/.test(adminIndexText)) {
    throw new Error('admin editor must not keep Rich Text image insertion guards after disabling Rich Text.');
  }
  if (!/syncGlobalAdminNavVisibility/.test(adminIndexText) || !/data-cms-editor-global-nav/.test(adminIndexText)) {
    throw new Error('admin editor must hide the global CMS navigation on posts editor routes.');
  }
  if (!/MutationObserver\(scheduleEditorChromeSync\)/.test(adminIndexText) || !/hashchange', syncEditorChromeSoon/.test(adminIndexText)) {
    throw new Error('admin editor chrome sync must run across route changes and async CMS rerenders.');
  }

  console.log(`Admin checks passed: ${indexFiles.length} posts, ${actualDirs.length} archive dirs.`);
}

main();
