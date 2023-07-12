# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from torchutils import __version__
projects = 'ConvenienceTorch'
copyright = '2023, Nischal Bhattarai'
author = 'Nischal Bhattarai'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration"
    ,"sphinx.ext.autosectionlabel"
    ,"sphinx.ext.autodoc"
    ,"sphinx.ext.autosummary"
    ,"autoapi.extension"
    ,"sphinx.ext.viewcode"
    ,"sphinx.ext.napoleon"
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

## AutoApi Dirs
autoapi_type = 'python'
autoapi_dirs = ['../../src/torchutils']
