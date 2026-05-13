# 个人知识站（Eleventy）

目标：提供技术出版级阅读体验，并保持可持续在线编辑。

## 本地开发

- 安装依赖：`npm install`
- 开发模式：`npm run dev`
- 构建站点：`npm run build`

在 PowerShell 执行策略拦截 `npm.ps1` 时，使用 `npm.cmd run build` 或 `npm.cmd run dev`。

默认 Eleventy 输出目录：`_site/`。

## 目录结构

- `src/index.njk`：首页
- `src/posts/index.njk`：归档页
- `src/about.njk`：关于页
- `src/_layouts/base.njk`：全站框架、导航、搜索、页脚
- `src/_layouts/post.njk`：文章页布局、目录、评论、阅读增强
- `src/css/style.css`：全站样式
- `src/js/main.js`：搜索、移动菜单、主题切换、灯箱、代码复制
- `src/admin/config.yml`：Decap CMS 内容模型
- `src/admin/index.html`：后台入口、写作说明、预览初始化

文章内容位于：`src/posts/**/index.md`。

## 在线编辑（Decap CMS）

- 后台入口：`/admin`
- 登录后可创建草稿（Editorial Workflow）、预览并发布
- 保留 GitHub backend + OAuth 接入（`/api/auth`）
- Admin 回归检查清单：`docs/admin-regression-checklist.md`

### OAuth 实现与部署建议

- 主方案（推荐）：Cloudflare Pages Functions
- 使用 `functions/api/auth.js` + `functions/api/callback.js` 提供 `/api/auth` 与 `/api/callback`。
- `src/admin/config.yml` 中保持 `auth_endpoint: /api/auth` 不变。
- 需配置环境变量：`GITHUB_CLIENT_ID`、`GITHUB_CLIENT_SECRET`，可选 `AUTH_BASE_URL`（用于固定回调域名）。
- 备用方案：`oauth-worker/` 独立 Worker。仅在无法使用 Pages Functions 时启用，避免两套 OAuth 同时对外暴露造成漂移。

### 新建归档目录

- `archive` 字段使用 `archive-combobox`，下拉只合并三类来源并做严格清洗：
- `src/posts` 下真实一级目录（由 `/admin-archives.json` 生成）。
- `config.yml` 中显式 `options`（同样会被清洗）。
- 本地历史（`localStorage` 的 `admin_archive_history_v2`，旧 `v1` 会忽略）。
- 输入的新目录会用于生成文章路径：`src/posts/<archive>/<slug>/index.md`。
- 目录名要求：非空、无 `/` 或 `\`、非 `.md/index.md`、非 `[object Object]`，仅允许中文/字母/数字/`_`/`-`。

### 已发布文章移动归档目录

- Decap 默认仅更新 front matter 的 `archive` 字段，不会移动既有文件路径。
- `/admin` 内置“移动文章归档”面板（仅在编辑已有文章且路径可识别时显示）。
- 面板读取当前 entry 路径，并按 `src/posts/<targetArchive>/<slug>/` 目录迁移。
- 使用当前登录 GitHub token 调用 Contents API：递归复制该文章目录内所有文件（含 `index.md`、`img/**`、`video/**`）到新目录，再删除旧目录文件。
- 对大于 1MB 导致 Contents API `content` 不可用的文件，迁移逻辑会自动回退到 Git Blobs API 按 `sha` 读取 base64 内容。
- 若目标目录任一文件已存在会中止迁移并报错，避免覆盖。
- 成功后提示刷新并从新路径继续编辑。
- 仍受 GitHub API 与仓库单文件大小限制约束；超限文件无法通过该流程迁移。

### Markdown / HTML / 公式 / 脚注

- 编辑器正文字段支持 Markdown。
- Eleventy 与后台预览均启用 `markdown-it` 的 `html: true`，允许合法内联/块级 HTML。
- 后台预览支持：Markdown、HTML、脚注、KaTeX、代码块、表格、图片。

### 媒体上传

- posts 集合配置：`media_folder: img`、`public_folder: ./img`。
- 新建/编辑文章时，图片会写入该文章目录下的 `img` 文件夹，并以 `./img/...` 引用。
- 若后台环境或插件对动态目录行为存在差异，可手动确认引用路径为 `./img/<filename>`。

### 草稿与发布

- 发布模式：`publish_mode: editorial_workflow`。
- 推荐流程：`New` 新建 -> 保存草稿 -> 预览检查 -> `Publish` 发布到 GitHub。

## 发布前检查

- 构建检查：`npm.cmd run build`
- Admin 一致性检查：`npm.cmd run check:admin`（需先 build 生成 `_site/admin-archives.json`）
- 本地预览：`npm.cmd run start`
