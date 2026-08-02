"""Fail if any formula in the built site would not be typeset.

``mkdocs build --strict`` fails on a broken link and has no opinion about
whether maths rendered, and neither did anything else: the tutorial published
every one of its formulas as its own LaTeX source for as long as the page
existed, and the first attempt at fixing it changed nothing while a test said
it had. This walks the built HTML the way MathJax does and reports what it
would skip.

The walk is transcribed from the vendored bundle's ``handleContainer``:

    o = processHtmlClass.exec(class)
    if (!firstChild || data-MJX || (!o && skipHtmlTags.exec(tag))) skip subtree
    else ignore = (ignore || ignoreHtmlClass.exec(class)) && !o

The second line is the one that matters. The flag is re-evaluated at every
element, so an exemption granted on an ancestor is revoked by any descendant
that matches ignoreHtmlClass -- which is why a catch-all ignore pattern can
never reach maths sitting several divs down, and that is where nbconvert puts
it.

Being ignored is not by itself a failure: notebook code cells are ignored
deliberately. What fails is maths inside a *markdown* cell or an arithmatex
span that would not be typeset.

    python mkdocs-documentation/checks/check_rendered_maths.py [site] [mathjax.js]
"""

import pathlib
import re
import sys
from html.parser import HTMLParser

SKIP_TAGS = {
    "script",
    "noscript",
    "style",
    "textarea",
    "pre",
    "code",
    "annotation",
    "annotation-xml",
}
VOID = {"br", "img", "hr", "meta", "link", "input", "source", "col"}

# Where a reader is entitled to see rendered maths.
RENDERED = ("jp-MarkdownCell", "arithmatex")
MATHS = re.compile(r"\$\$|\$[^$\n]+\$|\\\(|\\\[")


def patterns(ignore_class, process_class):
    """Return the two regexes MathJax builds from the options.

    Parameters
    ----------
    ignore_class : str
        The ``ignoreHtmlClass`` option.
    process_class : str
        The ``processHtmlClass`` option.

    Returns
    -------
    tuple of re.Pattern
        The ignore pattern and the process pattern.
    """
    return (
        re.compile(r"(?:^| )(?:" + ignore_class + r")(?: |$)"),
        re.compile(r"(?:^| )(?:" + process_class + r")(?: |$)"),
    )


class Walk(HTMLParser):
    """Track the ignore flag exactly as MathJax's handleContainer does."""

    def __init__(self, ignore_re, process_re):
        super().__init__(convert_charrefs=True)
        self.ignore_re, self.process_re = ignore_re, process_re
        self.ignored = [False]
        self.rendered = [False]
        self.skipping = 0
        self.typeset = []
        self.lost = []

    def handle_starttag(self, tag, attrs):
        """Push the flags this element leaves to its children."""
        if tag in VOID:
            return
        cls = dict(attrs).get("class", "") or ""
        process = bool(self.process_re.search(cls))
        if self.skipping or (not process and tag in SKIP_TAGS):
            self.skipping += 1
            self.ignored.append(self.ignored[-1])
            self.rendered.append(self.rendered[-1])
            return
        self.ignored.append(
            (self.ignored[-1] or bool(self.ignore_re.search(cls))) and not process
        )
        self.rendered.append(
            self.rendered[-1] or any(name in cls.split() for name in RENDERED)
        )

    def handle_endtag(self, tag):
        """Pop them again."""
        if tag in VOID:
            return
        if self.skipping:
            self.skipping -= 1
        if len(self.ignored) > 1:
            self.ignored.pop()
            self.rendered.pop()

    def handle_data(self, data):
        """Record any maths, and whether it would be typeset."""
        if self.skipping or not MATHS.search(data):
            return
        text = data.strip()[:58]
        if not self.ignored[-1]:
            self.typeset.append(text)
        elif self.rendered[-1]:
            self.lost.append(text)


def config(js_path):
    """Return ``(ignoreHtmlClass, processHtmlClass)`` as written in mathjax.js.

    Parameters
    ----------
    js_path : pathlib.Path
        The MathJax configuration file.

    Returns
    -------
    tuple of str
        The two options.
    """
    text = pathlib.Path(js_path).read_text(encoding="utf-8")
    # Only from inside the options object: the comment above it quotes the old
    # values in order to explain them, and reading those was this script's
    # first bug.
    block = re.search(r"options:\s*\{(.*?)\}", text, re.S)
    if not block:
        sys.exit(f"no options block in {js_path}")
    body = block.group(1)
    ignore = re.search(r'ignoreHtmlClass:\s*"([^"]*)"', body)
    process = re.search(r'processHtmlClass:\s*"([^"]*)"', body)
    if not (ignore and process):
        sys.exit(f"could not read the options out of {js_path}")
    return ignore.group(1), process.group(1)


def main():
    """Walk every built page and fail on maths that would not be typeset."""
    site = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else "mkdocs-documentation/site"
    )
    js = pathlib.Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "mkdocs-documentation/docs/javascripts/mathjax.js"
    )
    ignore_class, process_class = config(js)
    print(f"ignoreHtmlClass={ignore_class!r}  processHtmlClass={process_class!r}")

    pages = sorted(site.rglob("*.html"))
    if not pages:
        sys.exit(f"no built pages under {site}; build the site first")

    total = 0
    lost_pages = []
    for page in pages:
        walk = Walk(*patterns(ignore_class, process_class))
        walk.feed(page.read_text(encoding="utf-8"))
        total += len(walk.typeset)
        if walk.lost:
            lost_pages.append((page.relative_to(site), walk.lost))

    if lost_pages:
        lines = ["maths MathJax would walk past, leaving the source on the page:", ""]
        for rel, lost in lost_pages:
            lines.append(f"  {rel}: {len(lost)}")
            lines += [f"      {text!r}" for text in lost[:3]]
        sys.exit("\n".join(lines))

    if not total:
        sys.exit(
            f"walked {len(pages)} pages and found no maths at all. Either the "
            f"renderer changed its markup or the site is not built; either way "
            f"this check is no longer checking anything."
        )
    print(f"{len(pages)} pages, {total} formulas, none skipped")


if __name__ == "__main__":
    main()
