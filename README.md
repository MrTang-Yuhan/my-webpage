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

### 新建归档目录

- `archive` 字段为自由输入（`string`），不再限制固定下拉。
- 输入的新目录会用于生成文章路径：`src/posts/<archive>/<slug>/index.md`。
- 建议命名：小写英文/数字/连字符，或中文目录名，避免空格和特殊符号。

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
- 本地预览：`npm.cmd run start`
