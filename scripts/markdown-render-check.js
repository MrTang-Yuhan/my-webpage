const MarkdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");
const markdownItKatex = require("@vscode/markdown-it-katex").default;
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const katex = require("katex");

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

function runChecks(label, html) {
  const checks = [
    ["inline math renders", /<span class="katex">/.test(html)],
    ["block math renders", /class="katex-block"/.test(html) || /class="katex-display"/.test(html)],
    ["aligned environment renders", /annotation encoding="application\/x-tex">[\s\S]*\\begin\{aligned\}/.test(html)],
    ["table math renders", /<td><span class="katex">/.test(html)],
    ["footnote renders", /<section class="footnotes">/.test(html) && /Footnote body/.test(html)],
    ["code fence stays literal", /const value = &quot;\$not_math\$&quot;;/.test(html)]
  ];

  return checks
    .filter(([, ok]) => !ok)
    .map(([name]) => `${label}: ${name}`);
}

function loadAdminMarkdownIt() {
  const adminPath = path.join(__dirname, "..", "src", "admin", "index.html");
  const adminIndex = fs.readFileSync(adminPath, "utf8");
  const match = adminIndex.match(/function registerVsCodeMarkdownItKatex\(md, options\) \{[\s\S]*?\n        \}\n\n        const md =/);
  if (!match) {
    throw new Error("Admin VS Code KaTeX plugin function was not found.");
  }

  const functionSource = match[0].replace(/\n\n        const md =$/, "");
  const sandbox = {
    window: { katex },
    console
  };
  vm.createContext(sandbox);
  vm.runInContext(functionSource, sandbox);

  const adminMd = MarkdownIt({
    html: true,
    breaks: false,
    linkify: true
  });
  adminMd.use(markdownItFootnote);
  sandbox.registerVsCodeMarkdownItKatex(adminMd, { throwOnError: false, strict: "ignore" });
  return adminMd;
}

const failed = [
  ...runChecks("Eleventy markdown", md.render(sample)),
  ...runChecks("Admin preview markdown", loadAdminMarkdownIt().render(sample))
];

if (failed.length) {
  console.error("Markdown render checks failed:");
  failed.forEach((name) => console.error("- " + name));
  process.exit(1);
}

console.log("Markdown render checks passed.");
