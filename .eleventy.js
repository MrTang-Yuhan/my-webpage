const markdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");

module.exports = function(eleventyConfig) {
  const md = markdownIt({
    html: true,
    breaks: false,
    linkify: true
  }).use(markdownItFootnote);
  eleventyConfig.setLibrary("md", md);

  // Pass through static files
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");
  eleventyConfig.addPassthroughCopy("src/images");
  eleventyConfig.addPassthroughCopy("src/admin/config.yml");
  eleventyConfig.addPassthroughCopy("src/posts/**/img");
  eleventyConfig.addPassthroughCopy("src/posts/**/video");

  // Add filter for formatting dates
  eleventyConfig.addFilter("formatDate", function(date) {
    const d = new Date(date);
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
