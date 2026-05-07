# Blog

A minimalist static blog built with [Eleventy](https://www.11ty.dev/).

## Features

- Clean, minimalist design (similar to siboehm.com)
- Markdown-based blog posts
- Each blog in a separate subdirectory
- Auto-updates to homepage when new blogs are added
- Navigation
- Search functionality
- GitHub comments via Giscus
- Dark mode support
- Responsive design
- Local preview
- Deployable to GitHub Pages and Vercel

## Quick Start

### Install dependencies

```bash
npm install
```

### Local preview

```bash
npm start
```

The site will be available at `http://localhost:8080`

### Build for production

```bash
npm run build
```

The built files will be in `_site/` directory.

## Creating New Blog Posts

1. Create a new directory under `src/posts/` (e.g., `src/posts/my-new-post/`)
2. Create an `index.md` file with the following frontmatter:

```markdown
---
layout: post.njk
title: "Your Post Title"
date: 2026-05-07
description: "A brief description of your post."
tags:
  - post
  - your tags
---

Your content here...
```

3. The post will automatically appear on the homepage and archive page after rebuilding.

### Adding Footnotes (Margin Notes)

Use the `<aside class="footnote">` HTML element for margin footnotes:

```markdown
<aside class="footnote">
  <p>Your footnote content here.</p>
</aside>
```

Footnotes float to the right on desktop screens, matching the siboehm.com style.

### Adding Images

Use standard Markdown syntax:

```markdown
![Alt text](https://picsum.photos/seed/your-image/800/400)
```

Or use the HTML `<img>` tag for more control:

```markdown
<img src="https://picsum.photos/seed/your-image/800/400" alt="Description">
```

Random images available at [picsum.photos](https://picsum.photos/).

## Configuration

### GitHub Comments (Giscus)

To enable GitHub comments:

1. Go to [giscus.app](https://giscus.app) to set up and get your repo ID
2. Update `src/_layouts/post.njk` with your giscus configuration:
   - `data-repo`: Your GitHub repo (e.g., `username/repo`)
   - `data-repo-id`: Your repo ID
   - `data-category` and `data-category-id`: Your category settings

### Site Metadata

Edit the templates to update site-wide metadata like author name, social links, etc.

## Deployment

### GitHub Pages

1. Push your code to a GitHub repository
2. Enable GitHub Pages in repository settings
3. The GitHub Actions workflow will automatically deploy on push to main

### Vercel

1. Import your repository to Vercel
2. Vercel will auto-detect the `vercel.json` configuration
3. Deploy!

Or use the Vercel CLI:

```bash
npm i -g vercel
vercel
```

## Project Structure

```
├── .eleventy.js          # Eleventy configuration
├── vercel.json           # Vercel configuration
├── package.json
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions workflow
└── src/
    ├── _includes/        # Reusable templates
    ├── _layouts/          # Page layouts
    ├── css/
    │   └── style.css     # Styles
    ├── js/
    │   └── main.js       # JavaScript
    ├── posts/            # Blog posts (one subdirectory per post)
    │   ├── blog-1/
    │   │   └── index.md
    │   └── blog-2/
    │       └── index.md
    ├── index.njk         # Homepage
    ├── about.njk         # About page
    └── search-index.njk  # Search index generator
```

## License

MIT
