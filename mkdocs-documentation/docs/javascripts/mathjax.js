// Two kinds of page reach MathJax here, and they arrive in different shapes.
//
// The .md pages go through pymdownx.arithmatex, which rewrites $...$ into
// \(...\) and wraps the result in class="arithmatex". That is what the
// ignore/process pair below is for: ignore the whole document, process that
// class, and nothing in a code block is ever mistaken for maths.
//
// nlse_tutorial.ipynb does not. mkdocs-jupyter renders it with nbconvert,
// which is not the mkdocs markdown pipeline, so arithmatex never sees it: the
// maths lands in the page as literal $...$ and $$...$$ inside
// class="jp-MarkdownCell". Against the old configuration that failed twice
// over -- the class was not processed and the delimiters were not recognised
// -- so every formula in the tutorial was served as its own source, 6 display
// and 22 inline.
//
// So: process that class as well, and accept the dollar delimiters. The
// delimiters are safe to add because they are only ever looked for inside the
// two processed classes. Notebook *code* cells are jp-CodeCell, a sibling
// rather than a child, so a shell prompt or a format string in the tutorial
// is still left alone.
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|jp-MarkdownCell",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
