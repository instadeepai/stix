# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Configuration file for the Sphinx documentation builder.


import tomllib
from pathlib import Path

project = "stix"
author = "InstaDeep"

_pyproject = tomllib.loads(
    (Path(__file__).resolve().parent.parent.parent / "pyproject.toml").read_text()
)
release = _pyproject["project"]["version"]
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]
default_role = "code"

# Napoleon settings (Google-style docstrings, matching the ruff convention).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc settings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports: list[str] = []
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # Document ``__call__`` — the primary public API of the callable classes
    # (networks, solvers, criteria); autodoc skips dunders by default.
    "special-members": "__call__",
}

# Cross-project references.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
}

exclude_patterns: list[str] = []

html_theme = "furo"
html_title = "STIX"
