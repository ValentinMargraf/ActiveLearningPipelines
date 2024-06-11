import os
import sys

import commonmark
from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../ALP"))

import ALP

# -- Read the Docs ---------------------------------------------------------------------------------
master_doc = "index"

# -- Project information ---------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "ALP"
copyright = "2024, Margraf et al."
author = "Margraf et al."
release = ALP.__version__
version = ALP.__version__

# -- General configuration -------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "nbsphinx",
    "sphinx.ext.duration",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.more_autodoc.autoprotocol",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}


# -- Options for HTML output -----------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
pygments_dark_style = "monokai"
html_theme_options = {"sidebar_hide_name": True}
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "class"
autodoc_inherit_docstrings = False

# -- Images ----------------------------------------------------------------------------------------
StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]
# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Markdown in docstring -----------------------------------------------------------------------------
# https://gist.github.com/dmwyatt/0f555c8f9c129c0ac6fed6fabe49078b#file-docstrings-py
# based on https://stackoverflow.com/a/56428123/23972


def docstring(app, what, name, obj, options, lines):
    if len(lines) > 1 and lines[0] == "@&ismd":
        md = "\n".join(lines[1:])
        ast = commonmark.Parser().parse(md)
        rst = commonmark.ReStructuredTextRenderer().render(ast)
        lines.clear()
        lines += rst.splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", docstring)
