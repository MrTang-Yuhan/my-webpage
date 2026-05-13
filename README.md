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
- `src/admin/index.html`：后台入口与预览初始化

文章内容位于：`src/posts/**/index.md`。

## 在线编辑（Decap CMS）

- 后台入口：`/admin`
- 登录后可创建草稿（Editorial Workflow）、预览并发布
- 文章会按配置写入：`src/posts/<archive>/<slug>/index.md`
- 文章图片默认进入对应文章目录下的 `img` 文件夹，并用 `./img` 路径引用

## 发布前检查

- 构建检查：`npm.cmd run build`
- 本地预览：`npm.cmd run start`
