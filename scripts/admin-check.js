const fs = require('fs');
const path = require('path');

const root = process.cwd();
const postsRoot = path.join(root, 'src', 'posts');
const adminArchivesPath = path.join(root, '_site', 'admin-archives.json');
const adminConfigPath = path.join(root, 'src', 'admin', 'config.yml');
const adminIndexPath = path.join(root, 'src', 'admin', 'index.html');
const legacyArchiveAliases = ['mode-parallelism'];

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

function readFrontMatterValue(filePath, key) {
  const text = fs.readFileSync(filePath, 'utf8');
  const fm = text.match(/^\uFEFF?---\r?\n([\s\S]*?)\r?\n---/);
  if (!fm) return '';
  const pattern = new RegExp(`^${key}:\\s*(.*)\\s*$`, 'm');
  const line = fm[1].match(pattern);
  return line ? line[1].trim() : '';
}

function readFrontMatterArchive(filePath) {
  return readFrontMatterValue(filePath, 'archive');
}

function readFrontMatterPostId(filePath) {
  return readFrontMatterValue(filePath, 'post_id');
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
  const postIdMismatches = [];
  const legacyArchiveRefs = [];
  for (const file of indexFiles) {
    const rel = normalizeSlash(path.relative(root, file));
    const m = rel.match(/^src\/posts\/([^/]+)\/[^/]+\/index\.md$/);
    if (!m) continue;
    const expectedArchive = m[1];
    const expectedPostId = rel.split('/')[3];
    const actualArchive = readFrontMatterArchive(file);
    const actualPostId = readFrontMatterPostId(file);
    if (legacyArchiveAliases.includes(actualArchive)) {
      legacyArchiveRefs.push({ file: rel, archive: actualArchive });
    }
    if (actualArchive !== expectedArchive) {
      mismatches.push({ file: rel, expectedArchive, actualArchive });
    }
    if (actualPostId !== expectedPostId) {
      postIdMismatches.push({ file: rel, expectedPostId, actualPostId });
    }
  }

  if (mismatches.length) {
    console.error('Archive mismatch detected:');
    for (const item of mismatches) {
      console.error(`- ${item.file}: front matter archive="${item.actualArchive}" expected="${item.expectedArchive}"`);
    }
    process.exit(1);
  }
  if (legacyArchiveRefs.length) {
    console.error('Legacy archive aliases are not allowed in post front matter:');
    for (const item of legacyArchiveRefs) {
      console.error(`- ${item.file}: archive="${item.archive}"`);
    }
    process.exit(1);
  }
  if (postIdMismatches.length) {
    console.error('Post ID mismatch detected:');
    for (const item of postIdMismatches) {
      console.error(`- ${item.file}: front matter post_id="${item.actualPostId}" expected="${item.expectedPostId}"`);
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
  for (const legacyArchive of legacyArchiveAliases) {
    if (new RegExp(`\\b${legacyArchive}\\b`).test(configText)) {
      throw new Error(`admin config must not contain legacy archive alias "${legacyArchive}".`);
    }
  }
  if (!/name:\s*body[\s\S]*?widget:\s*markdown[\s\S]*?modes:\s*\n\s*-\s*raw\b/.test(configText)) {
    throw new Error('posts body markdown widget must keep raw mode enabled.');
  }
  if (!/identifier_field:\s*post_id\b/.test(configText)) {
    throw new Error('posts collection must use post_id as the stable identifier field.');
  }
  if (!/summary:\s*["']\{\{title\}\}["']/.test(configText)) {
    throw new Error('posts collection summary must display the title in the admin list.');
  }
  if (!/name:\s*post_id[\s\S]*?widget:\s*hidden\b/.test(configText)) {
    throw new Error('posts post_id field must stay hidden in the editor.');
  }
  if (!/path:\s*["']\{\{archive\}\}\/\{\{slug\}\}\/index["']/.test(configText)) {
    throw new Error('posts path must use archive plus stable identifier-derived slug.');
  }
  if (!/slug:\s*["']\{\{slug\}\}["']/.test(configText)) {
    throw new Error('posts slug must not include date fields.');
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
  if (!/name:\s*image_upload[\s\S]*?widget:\s*image[\s\S]*?media_folder:\s*img\b/.test(configText)) {
    throw new Error('posts editor must expose a Decap image upload widget scoped to the current post img directory.');
  }
  if (!/findImageUploadControl/.test(adminIndexText)) {
    throw new Error('admin upload-only image button must open the Decap image upload widget.');
  }
  if (!/name:\s*attachment_upload[\s\S]*?widget:\s*file[\s\S]*?media_folder:\s*attach\b/.test(configText)) {
    throw new Error('posts editor must expose a Decap file upload widget scoped to the current post attach directory.');
  }
  if (!/upload-attachment-only-btn/.test(adminIndexText) || !/findAttachmentUploadControl/.test(adminIndexText)) {
    throw new Error('admin editor must expose an upload-only attachment button that opens the Decap file widget.');
  }
  if (!/delete\('image_upload'\)/.test(adminIndexText) || !/delete\('attachment_upload'\)/.test(adminIndexText)) {
    throw new Error('admin editor must clear temporary upload fields before saving.');
  }
  if (!/findAddComponentToolbarAnchor/.test(adminIndexText) || !/add\\s\*component/.test(adminIndexText)) {
    throw new Error('admin upload-only image button must mount next to the Add Component toolbar button.');
  }
  if (!/data-cms-editor-upload-toolbar/.test(adminIndexText)) {
    throw new Error('admin upload-only image button toolbar must stay sticky while editing long posts.');
  }
  if (!/startUploadOnlyMarkdownGuard/.test(adminIndexText)) {
    throw new Error('admin upload-only image button must guard the Markdown body from any edits.');
  }
  if (!/isEditorBodyReady/.test(adminIndexText)) {
    throw new Error('admin editor tools must wait for the editor body before showing custom controls.');
  }
  if (!/getCachedArchiveDirs/.test(adminIndexText)) {
    throw new Error('admin archive directory lookup must be cached to avoid slow repeated editor sync fetches.');
  }
  if (/protectMarkdownSourceForRichTextImages|restoreMarkdownSourceExceptNewImages|beginRichTextImageSession|richTextActive/.test(adminIndexText)) {
    throw new Error('admin editor must not keep Rich Text image insertion guards after disabling Rich Text.');
  }
  if (!/syncGlobalAdminNavVisibility/.test(adminIndexText) || !/data-cms-editor-global-nav/.test(adminIndexText)) {
    throw new Error('admin editor must hide the global CMS navigation on posts editor routes.');
  }
  if (!/registerEventListener\(\{\s*name:\s*'preSave'/.test(adminIndexText) || !/post_id/.test(adminIndexText)) {
    throw new Error('admin editor must auto-generate hidden post_id before saving new posts.');
  }
  if (!/MutationObserver\(scheduleEditorChromeSync\)/.test(adminIndexText) || !/hashchange', syncEditorChromeSoon/.test(adminIndexText)) {
    throw new Error('admin editor chrome sync must run across route changes and async CMS rerenders.');
  }

  console.log(`Admin checks passed: ${indexFiles.length} posts, ${actualDirs.length} archive dirs.`);
}

main();
