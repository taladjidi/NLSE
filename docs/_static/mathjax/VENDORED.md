# MathJax, vendored

MathJax 3.2.2, from the `mathjax` npm package, pruned to what a page here
actually loads: the combined `tex-mml-chtml` bundle and the woff-v2 fonts it
fetches at run time. The fonts must stay at `output/chtml/fonts/woff-v2`
relative to the bundle, which is where it resolves them from.

It is *here* rather than loaded from a CDN because a `<script>` tag pointed at
someone else's host is an agreement to run whatever that host serves, in the
browser of everyone who reads these docs. That is not a theoretical objection:
the previous entry in `extra_javascript` was `polyfill.io`, which was sold in
2024 and began serving malware. Vendoring turns the dependency into something
that was reviewed once, at a version, in a commit.

Provenance, for anyone updating it:

    npm pack mathjax@3
    sha256  1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09
            mathjax-3.2.2.tgz

Apache-2.0; see LICENSE beside this file.
