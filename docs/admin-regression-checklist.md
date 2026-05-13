# Admin 回归检查清单

目标：验证 `/admin` 是否可以稳定完成一篇文章的写作闭环（新建、编辑、预览、保存、发布）。

## 1. 登录与导航

- 打开 `/admin`，可正常进入 Decap CMS。
- GitHub OAuth 登录成功，页面无空白或报错。
- 后台首页 `Contents / Workflow / Media` 导航可见且可点击。
- OAuth 路由使用 `/api/auth`（Pages Functions 主方案），回调为同域 `/api/callback` 或 `AUTH_BASE_URL` 指定域名。
- 登录后 callback 应写入 `admin_session`（HttpOnly/Secure/SameSite=Lax）会话 cookie。

## 2. 进入文章编辑器

- 从 Contents 进入 posts 集合，点击 `New` 新建文章。
- 新建/编辑文章页面中，`Contents / Workflow / Media` 横栏应隐藏。
- 编辑器主体、字段区、保存按钮、预览切换可见。
- 返回集合列表后，后台导航恢复显示。

## 3. 字段与归档目录

- `archive` 下拉只应出现干净目录名，不应出现 `index.md`、`.md`、`[object Object]`、带 `/` 的值或乱码片段。
- `archive` 可输入新目录名（非法值应被过滤，不进入下拉历史）。
- 标题、日期、摘要、标签可编辑并保存。
- 新文章路径符合 `src/posts/<archive>/<slug>/index.md`。
- `/admin-archives.json` 内容应仅为 `src/posts` 真实一级目录列表（JSON 字符串数组）。

## 3.1 已发布文章归档移动

- 编辑已有文章时应出现“移动文章归档”面板，并显示当前 `src/posts/<archive>/<slug>/index.md` 路径。
- 输入目标 archive 后点击移动，成功后应调用 `POST /api/admin/move-post` 并在 GitHub 看到单次 commit 完成目录迁移（至少包含 `index.md`，如存在 `img/`、`video/` 也应迁移）。
- 移动后重新打开条目时路径应更新到 `src/posts/<targetArchive>/<slug>/index.md`。
- 服务端需配置 `GITHUB_ADMIN_TOKEN`；建议配置 `ADMIN_SESSION_SECRET`。缺失写 token 时接口应返回明确错误。
- 未登录（无 session）/会话过期/跨域请求时，接口应拒绝并返回明确错误。

## 4. Markdown / HTML / 公式 / 脚注

- Markdown 标题、列表、引用、代码块显示正常。
- 合法内联/块级 HTML 在预览中可渲染。
- `$...$` 与 `$$...$$`（含多行块写法）可渲染。
- 脚注可渲染，且预览不报错。

## 5. 图片与媒体

- 上传图片成功，无权限或路径错误。
- 图片引用路径为 `./img/<filename>`。
- 预览中图片可显示。
- 对含大媒体（>1MB）的文章执行归档迁移时应成功（服务端迁移通过 Git tree/blob 方式，不依赖浏览器 Contents API 文件内容）。

## 6. 草稿与发布

- 保存草稿成功（Editorial Workflow）。
- 文章可从 Draft 流转到 Publish。
- 发布后可返回列表，状态与条目显示正常。

## 7. 回归与构建

- 执行 `npm.cmd run build` 成功。
- 执行 `npm.cmd run check:admin` 成功（archive/front matter 一致、`admin-archives.json` 可解析且目录集合一致）。
- 不修改文章正文的前提下，前台构建不受 Admin 逻辑影响。
