const MarkdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");
const markdownItKatex = require("@vscode/markdown-it-katex").default;

const md = MarkdownIt({
  html: true,
  breaks: false,
  linkify: true
})
  .use(markdownItFootnote)
  .use(markdownItKatex, { throwOnError: false, strict: "ignore" });

const sample = [
  "Inline math: $x$.",
  "",
  "$$",
  "\\begin{aligned}",
  "a &= b\\\\",
  "c &= d",
  "\\end{aligned}",
  "$$",
  "",
  "| term | math |",
  "| --- | --- |",
  "| inline | $x^2$ |",
  "",
  "Footnote ref[^1].",
  "",
  "[^1]: Footnote body.",
  "",
  "```js",
  "const value = \"$not_math$\";",
  "```"
].join("\n");

const html = md.render(sample);

const checks = [
  ["inline math renders", /<span class="katex">/.test(html)],
  ["block math renders", /class="katex-block"/.test(html) || /class="katex-display"/.test(html)],
  ["aligned environment renders", /annotation encoding="application\/x-tex">[\s\S]*\\begin\{aligned\}/.test(html)],
  ["table math renders", /<td><span class="katex">/.test(html)],
  ["footnote renders", /<section class="footnotes">/.test(html) && /Footnote body/.test(html)],
  ["code fence stays literal", /const value = &quot;\$not_math\$&quot;;/.test(html)]
];

const failed = checks.filter(([, ok]) => !ok);

if (failed.length) {
  console.error("Markdown render checks failed:");
  failed.forEach(([name]) => console.error("- " + name));
  process.exit(1);
}

console.log("Markdown render checks passed.");
