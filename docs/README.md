# Documentation with Sphinx

## Install dependencies
`pip install -U recommonmark sphinx sphinx_rtd_theme sphinx_markdown_tables`

## Generate curated lists only
## Automatically generate module documentations
`rm -rf source && sphinx-apidoc -o source/ ../`

## Customize documentation contents
Write or modify each documentation page as a markdown file.
Include the markdown file in `index.rst` `toctree`.
You may modify the landing page `index.rst` itself in reStructuredText format.

## Generate HTML documentation website
`cp ../README.md overview.md; make html`
You may ignore the consistency warning that README.md is not included in toctree. README.md is this file, the instruction to generate documentation website.
The website is generated in `_build/html`
