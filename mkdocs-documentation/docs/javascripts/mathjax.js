// Two renderers feed this site and they hand MathJax different things.
//
// A .md page goes through pymdownx.arithmatex, which rewrites $...$ into
// \(...\) and wraps the result in <span class="arithmatex">, with the maths
// as that span's own text. nlse_tutorial.ipynb does not: mkdocs-jupyter
// renders it with nbconvert, outside the mkdocs markdown pipeline, so the
// maths arrives as literal $...$ nested several divs deep:
//
//     div.jp-Cell.jp-MarkdownCell
//       div.jp-Cell-inputWrapper
//         div.jp-InputArea
//           div.jp-RenderedMarkdown
//             p        <- the maths is here
//
// That nesting is why `ignoreHtmlClass: ".*|"` cannot be used, and it is a
// sharper trap than it looks. MathJax re-evaluates the flag at every element:
//
//     ignore = (ignore || ignoreHtmlClass.test(class)) && !processHtmlClass.test(class)
//
// With ".*|" the ignore pattern matches every element including classless
// ones, so an exemption granted on an ancestor is revoked by the very next
// div below it. Naming jp-MarkdownCell in processHtmlClass therefore changed
// nothing: the exemption died one level down, four levels above the <p>.
// Only an element whose *own* text is the maths can be exempted that way,
// which is exactly what an arithmatex span is and what nbconvert never emits.
//
// So the catch-all goes. What actually keeps $ in code from being read as
// maths is skipHtmlTags, which already covers pre, code, script, style and
// textarea by default and needs no help. That leaves notebook code cells,
// whose outputs are not always inside those tags -- hence jp-CodeCell.
//
// tests/test_docs.py pins the delimiters and refuses a catch-all ignore
// pattern; the docs CI job walks the built HTML the same way MathJax does and
// checks that every formula comes out processed.
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
    ignoreHtmlClass: "jp-CodeCell",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
