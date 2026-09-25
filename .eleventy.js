const markdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");
const markdownItKatex = require("@vscode/markdown-it-katex").default;

function findMatchingSpanEnd(html, start) {
  const tagPattern = /<\/?span\b[^>]*>/gi;
  tagPattern.lastIndex = start;
  let depth = 0;
  let match;
  while ((match = tagPattern.exec(html)) !== null) {
    if (/^<\//.test(match[0])) {
      depth -= 1;
      if (depth === 0) return tagPattern.lastIndex;
    } else if (!/\/\s*>$/.test(match[0])) {
      depth += 1;
    }
  }
  return -1;
}

function protectKatexMarkup(html) {
  const formulas = [];
  const openingPattern = /<span\b[^>]*class=["'][^"']*\bkatex(?:-display)?\b[^"']*["'][^>]*>/gi;
  let result = "";
  let cursor = 0;
  let match;
  while ((match = openingPattern.exec(html)) !== null) {
    const end = findMatchingSpanEnd(html, match.index);
    if (end < 0) continue;
    result += html.slice(cursor, match.index);
    const token = `\uE000${formulas.length}\uE001`;
    formulas.push(html.slice(match.index, end));
    result += token;
    cursor = end;
    openingPattern.lastIndex = end;
  }
  return { html: result + html.slice(cursor), formulas };
}

function restoreKatexMarkup(value, formulas) {
  return value.replace(/\uE000(\d+)\uE001/g, function(_, index) {
    return formulas[Number(index)] || "";
  });
}

function katexPreviewWeight(markup) {
  const annotation = markup.match(/<annotation\b[^>]*>([\s\S]*?)<\/annotation>/i);
  if (!annotation) return 1;

  // Use a bounded approximation of the source formula's visible width. The
  // rendered KaTeX markup is much larger than what the reader sees.
  const sourceLength = annotation[1].replace(/<[^>]*>/g, "").trim().length;
  return Math.max(4, Math.min(32, sourceLength));
}

function truncateRenderedHtml(content, length) {
  const protectedHtml = protectKatexMarkup(String(content || ""));
  const plainText = protectedHtml.html
    .replace(/<\/(?:p|h[1-6]|li|blockquote|pre|table|tr|div)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const limit = Math.max(0, Number(length) || 0);
  const tokenPattern = /\uE000(\d+)\uE001/g;
  const output = [];
  let cursor = 0;
  let used = 0;
  let truncated = false;
  let match;

  while ((match = tokenPattern.exec(plainText)) !== null) {
    const before = plainText.slice(cursor, match.index);
    if (used + before.length > limit) {
      output.push(before.slice(0, Math.max(0, limit - used)));
      truncated = true;
      break;
    }
    output.push(before);
    used += before.length;

    const formulaIndex = Number(match[1]);
    const formula = protectedHtml.formulas[formulaIndex] || "";
    const weight = katexPreviewWeight(formula);
    if (used > 0 && used + weight > limit) {
      truncated = true;
      break;
    }
    output.push(match[0]);
    used += weight;
    cursor = match.index + match[0].length;
  }

  if (!truncated) {
    const tail = plainText.slice(cursor);
    if (used + tail.length > limit) {
      output.push(tail.slice(0, Math.max(0, limit - used)));
      truncated = true;
    } else {
      output.push(tail);
    }
  }

  let result = output.join("").trim();
  if (truncated) result = result.replace(/[\s。、，；：:：!?！？]+$/u, "") + "...";
  return restoreKatexMarkup(result, protectedHtml.formulas);
}

module.exports = async function(eleventyConfig) {
  const { default: markdownItShiki } = await import("@shikijs/markdown-it");

  const md = markdownIt({
    html: true,
    breaks: false,
    linkify: true
  })
    .use(markdownItFootnote)
    .use(markdownItKatex, { throwOnError: false, strict: "ignore" })
    .use(await markdownItShiki({
      theme: "github-light",
      langs: ["cpp", "python", "bash", "makefile", "markdown"],
      langAlias: {
        cuda: "cpp",
        cu: "cpp",
        sh: "bash"
      },
      fallbackLanguage: "text"
    }));
  eleventyConfig.setLibrary("md", md);

  eleventyConfig.addPreprocessor("rawCodeFences", "md", function(data, content) {
    return content.replace(
      /(^[ \t]{0,3}```[^\r\n]*(?:\r?\n)[\s\S]*?^[ \t]{0,3}```[ \t]*$)/gm,
      "{% raw %}\n$1\n{% endraw %}"
    );
  });

  // Pass through static files
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");
  eleventyConfig.addPassthroughCopy("src/images");
  eleventyConfig.addPassthroughCopy("logo");
  eleventyConfig.addPassthroughCopy("src/admin/config.yml");
  eleventyConfig.addPassthroughCopy("src/posts/**/img");
  eleventyConfig.addPassthroughCopy("src/posts/**/video");
  eleventyConfig.addPassthroughCopy("src/posts/**/attach");

  // Add filter for formatting dates
  eleventyConfig.addFilter("formatDate", function(date) {
    if (!date) return "";
    const d = new Date(date);
    if (Number.isNaN(d.getTime())) return String(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  });

  // Add filter for reading time
  eleventyConfig.addFilter("readingTime", function(content) {
    const wordsPerMinute = 200;
    const text = content.replace(/<[^>]*>/g, '');
    const wordCount = text.split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / wordsPerMinute);
    return `${readingTime} min read`;
  });

  // Add filter for truncating text
  eleventyConfig.addFilter("truncate", function(content, length) {
    const text = content.replace(/<[^>]*>/g, '');
    if (text.length <= length) return text;
    return text.substring(0, length).trim() + '...';
  });

  // Truncate rendered post previews while keeping complete KaTeX formula markup.
  eleventyConfig.addFilter("truncateRenderedHtml", truncateRenderedHtml);

  // Add IDs to headings for anchor links
  eleventyConfig.addFilter("addHeadingIds", function(content) {
    if (!content) return content;
    let counter = 0;
    return content.replace(/<(h[1-6])[^>]*>(.*?)<\/\1>/gi, function(match, tag, text) {
      const id = text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
      const uniqueId = id || 'heading-' + counter++;
      return `<${tag} id="${uniqueId}">${text}</${tag}>`;
    });
  });

  // Support simple video shortcode in markdown:
  // ::video{src="./video/demo.mp4" poster="./img/poster.png" caption="Demo"}
  eleventyConfig.addFilter("videoShortcode", function(content) {
    if (!content) return content;
    return content.replace(
      /<p>::video\{([^}]*)\}<\/p>/g,
      function(_, attrs) {
        const src = (attrs.match(/src="([^"]+)"/) || [])[1];
        const poster = (attrs.match(/poster="([^"]+)"/) || [])[1];
        const caption = (attrs.match(/caption="([^"]+)"/) || [])[1];
        if (!src) return _;
        const posterAttr = poster ? ` poster="${poster}"` : "";
        const captionHtml = caption ? `\n<p><em>${caption}</em></p>` : "";
        return `<video controls${posterAttr}><source src="${src}" type="video/mp4"></video>${captionHtml}`;
      }
    );
  });

  // Generate search index using a JavaScript template
  eleventyConfig.addCollection('searchablePosts', function(collectionApi) {
    return collectionApi.getFilteredByGlob('src/posts/**/*');
  });

  // Renumber footnotes sequentially
  eleventyConfig.addFilter("renumberFootnotes", function(content) {
    if (!content) return content;

    // Convert markdown-it-footnote output into the site's existing aside footnote format.
    // This keeps markdown footnotes compatible with current CSS and numbering logic.
    let normalized = content
      .replace(/<hr class="footnotes-sep">\s*/g, "")
      .replace(/<section class="footnotes">[\s\S]*?<\/section>/g, function(sectionHtml) {
        const items = [];
        const liPattern = /<li id="fn(\d+)" class="footnote-item">([\s\S]*?)<\/li>/g;
        let liMatch;
        while ((liMatch = liPattern.exec(sectionHtml)) !== null) {
          const id = liMatch[1];
          const rawBody = liMatch[2];
          const body = rawBody
            .replace(/<a href="#fnref\d+" class="footnote-backref">[\s\S]*?<\/a>/g, "")
            .trim();
          items.push(`<aside id="fn${id}" class="footnote">${body}</aside>`);
        }
        return items.join("\n");
      });

    // Collect all footnote refs in order they appear in content
    const refPattern = /<sup class="footnote-ref"><a href="#fn(\d+)"[^>]*>\[(\d+)\]<\/a><\/sup>/g;
    const refs = [];
    let match;
    while ((match = refPattern.exec(normalized)) !== null) {
      refs.push({ id: match[1], num: match[2] });
    }

    // Get unique footnote IDs in order of first appearance
    const footnoteIds = [];
    const idSet = new Set();
    refs.forEach(r => {
      if (!idSet.has(r.id)) {
        footnoteIds.push(r.id);
        idSet.add(r.id);
      }
    });

    // Create mapping from old ID to new sequential number
    const mapping = {};
    footnoteIds.forEach((id, index) => {
      mapping[id] = index + 1;
    });

    // Replace footnote refs with new numbers
    let result = normalized.replace(/<sup class="footnote-ref"><a href="#fn(\d+)"[^>]*>\[(\d+)\]<\/a><\/sup>/g, function(match, id, num) {
      const newNum = mapping[id];
      return `<sup class="footnote-ref"><a href="#fn${newNum}">[${newNum}]</a></sup>`;
    });

    // Replace footnote definitions with new numbers
    result = result.replace(/<aside id="fn(\d+)" class="footnote">/g, function(match, id) {
      const newNum = mapping[id];
      return `<aside id="fn${newNum}" class="footnote">`;
    });

    return result;
  });

  // Group posts by directory for archive
  eleventyConfig.addCollection('postsByDir', function(collectionApi) {
    const posts = collectionApi.getFilteredByGlob('src/posts/**/*').filter(item => item.data.layout === 'post.njk');

    // Check for duplicate titles
    const titles = {};
    posts.forEach(post => {
      const title = post.data.title;
      if (title) {
        if (titles[title]) {
          throw new Error(`Duplicate post title found: "${title}" in ${post.filePathStem} conflicts with ${titles[title].filePathStem}. Please rename one of them.`);
        }
        titles[title] = post;
      }
    });

    const grouped = {};
    posts.forEach(post => {
      const match = post.filePathStem.match(/\/posts\/([^/]+)/);
      if (match) {
        const dir = match[1];
        if (!grouped[dir]) grouped[dir] = [];
        grouped[dir].push(post);
      }
    });
    // Sort directories alphabetically, newest posts first within each
    Object.keys(grouped).sort().reverse().forEach(dir => {
      grouped[dir].sort((a, b) => new Date(b.date) - new Date(a.date));
    });
    return grouped;
  });

  // Clean top-level archive directory names under src/posts for admin usage.
  eleventyConfig.addCollection("adminArchiveDirs", function(collectionApi) {
    const posts = collectionApi.getFilteredByGlob("src/posts/**/*");

    const dirs = new Set();
    posts.forEach(post => {
      const normalizedPath = String(post.inputPath || "").replace(/\\/g, "/");
      const match = normalizedPath.match(/src\/posts\/([^/]+)\/[^/]+\/index\.md$/);
      if (!match) return;
      const dir = String(match[1] || "").trim();
      if (!dir) return;
      if (dir.includes("/") || dir.includes("\\")) return;
      if (/^index\.md$/i.test(dir)) return;
      dirs.add(dir);
    });

    return Array.from(dirs).sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
  });

  // Return configuration
  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      layouts: "_layouts",
      posts: "posts"
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk"
  };
};
