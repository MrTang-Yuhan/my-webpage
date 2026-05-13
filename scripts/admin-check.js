const fs = require('fs');
const path = require('path');

const root = process.cwd();
const postsRoot = path.join(root, 'src', 'posts');
const adminArchivesPath = path.join(root, '_site', 'admin-archives.json');

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

  console.log(`Admin checks passed: ${indexFiles.length} posts, ${actualDirs.length} archive dirs.`);
}

main();