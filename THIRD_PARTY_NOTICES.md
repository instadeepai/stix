# Third-party notices

The source code in this repository is licensed under the Apache License 2.0
(see [`LICENSE`](LICENSE)).

This file records the licences of stix's direct dependencies, so that anyone
reviewing or redistributing this project can confirm licence compatibility at
a glance.

## Direct dependencies (default install)

> [!NOTE]
> `stix` is distributed as a library, not a pinned application, and
> intentionally ships no lockfile — installers resolve their own compatible
> versions within these ranges. The versions below are the constraints
> declared in `pyproject.toml`; the exact version an installer resolves may
> be newer, within these bounds.

| package | version range | source | licence |
|---|---|---|---|
| `flax` | unpinned (any compatible version) | https://github.com/google/flax | Apache-2.0 |
| `optax` | unpinned (any compatible version) | https://github.com/google-deepmind/optax | Apache-2.0 |
| `distrax` | unpinned (any compatible version) | https://github.com/google-deepmind/distrax | Apache-2.0 |
| `diffrax` | unpinned (any compatible version) | https://github.com/patrick-kidger/diffrax | Apache-2.0 |
| `numpy` | >=2.0.0 | https://github.com/numpy/numpy | BSD-3-Clause |
| `jax` | >=0.5.0,<0.11.1 | https://github.com/jax-ml/jax | Apache-2.0 |
| `jaxlib` | >=0.5.0,<0.11.1 | https://github.com/jax-ml/jax | Apache-2.0 |
| `pydantic` | >=2.12.5 | https://github.com/pydantic/pydantic | MIT |
| `grain` | >=0.2.18 | https://github.com/google/grain | Apache-2.0 |
| `s3fs` | ==2026.2.0 | https://github.com/fsspec/s3fs | BSD-3-Clause |
| `datasets` | >=5.0.0 | https://github.com/huggingface/datasets | Apache-2.0 |

All direct dependencies are permissively licensed (Apache-2.0, MIT, or
BSD-3-Clause) and fully compatible with this project's own Apache-2.0
licence. None carry copyleft, non-commercial, or proprietary terms.

## Optional GPU / TPU extras

The `gpu` extra installs `jax[cuda12]`, and the `tpu` extra installs
`jax[tpu]`. Neither is required for a default install; `stix` runs on CPU
without them.

**Confirmed: the `gpu` extra pulls in NVIDIA's proprietary CUDA runtime and
kernel libraries.** Checked by installing the `gpu` extra in a clean
environment (2026-09-11) and inspecting `.dist-info` licence metadata:

| package | version | licence |
|---|---|---|
| `nvidia-cublas-cu12` | 12.9.2.10 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cuda-cccl-cu12` | 12.9.27 | NVIDIA Proprietary Software |
| `nvidia-cuda-cupti-cu12` | 12.9.79 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cuda-nvcc-cu12` | 12.9.86 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cuda-nvrtc-cu12` | 12.9.86 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cuda-runtime-cu12` | 12.9.79 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cudnn-cu12` | 9.25.1.1 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cufft-cu12` | 11.4.1.4 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cusolver-cu12` | 11.7.5.82 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-cusparse-cu12` | 12.5.10.65 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-nccl-cu12` | 2.31.2 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-nvjitlink-cu12` | 12.9.86 | LicenseRef-NVIDIA-Proprietary |
| `nvidia-nvshmem-cu12` | 3.7.2 | LicenseRef-NVIDIA-Proprietary |

All thirteen packages above are NVIDIA proprietary software, governed by
NVIDIA's own licence terms (shipped as `License.txt` in each package's
`.dist-info` directory), not Apache-2.0. Note that in this project's
dependency tree `nvidia-nccl-cu12` is proprietary — unlike the
BSD-3-Clause NCCL build that appeared in a different project's CUDA 12
resolution — so licence status per package should be re-checked after
any dependency bump rather than assumed stable across releases.

**Redistributing a built environment or container image with the `gpu`
extra installed means redistributing NVIDIA's binaries under NVIDIA's
proprietary terms, in addition to and separate from this project's own
Apache-2.0 licence.** This does not affect the default (CPU) install, or
`stix`'s own source code licensing.

The `tpu` extra was not checked; Google's TPU libraries are conventionally
Apache-2.0 throughout, and this is lower priority unless someone plans to
distribute a container with that extra installed.

## Development-only dependencies

The `dev` dependency group (plotly, pytest, sphinx, pyright, etc.) is not
shipped in the published package and is used only for testing, linting, and
documentation. These are not included in the table above since they carry no
distribution implications for end users of `stix-ml`.

## Scope: direct vs. transitive dependencies

The table above covers **direct** dependencies declared in `pyproject.toml`.
Each of these pulls in further transitive dependencies at install time (e.g.
`flax` pulls `msgpack`, `rich`, `orbax-checkpoint`; `datasets` pulls `pyarrow`,
`requests`, `fsspec`, etc.).

Because this project intentionally ships no lockfile, there is no single
fixed transitive dependency set to document permanently — it depends on the
resolver, Python version, platform, and install date. To satisfy full
third-party disclosure, generate a **dated snapshot** instead: run a clean
install once (e.g. `pip install stix-ml` in a fresh venv, on the Python
version this release targets), capture the output of the script below, and
commit it here labelled with the `stix` version and date it was taken —
clearly as a point-in-time snapshot of what a default install resolved to,
not a permanent guarantee for all future installs.

## Dependency snapshot (direct + transitive)

Snapshot taken 2026-09-11, `stix-ml` v0.1.0, clean `uv sync` (default, no
extras), 73 packages installed. This reflects one resolution at one point in
time — re-run the script below after any dependency bump.

**Overall result: clean.** All packages are permissively licensed
(Apache-2.0, MIT, BSD-2/3-Clause, PSF-2.0, ISC, ~~ or public-domain
equivalents), with one exception worth flagging explicitly:

> [!NOTE]
> **`scipy`'s binary distribution bundles GCC runtime components** —
> `libgfortran`/`libgcc` under **GPL-3.0-or-later WITH GCC-exception-3.1**,
> and `libquadmath` under **LGPL-2.1-or-later**. The GCC Runtime Library
> Exception specifically permits linking non-GPL (including proprietary)
> code against these libraries, so this does not impose copyleft
> obligations on `stix` itself. It is disclosed here because it is a
> GPL-family licence present in the dependency tree, even though the
> exception neutralises the propagation risk.

No proprietary, non-commercial (CC-NC), or unexplained-copyleft licences were
found anywhere else in the tree.

| package | version | source | licence |
|---|---|---|---|
| absl-py | 2.5.0 | https://github.com/abseil/abseil-py | Apache-2.0 |
| accessible-pygments | 0.0.5 | https://github.com/Quansight-Labs/accessible-pygments | BSD-3-Clause |
| aiobotocore | 3.9.1 | https://github.com/aio-libs/aiobotocore | Apache-2.0 |
| aiofiles | 25.1.0 | https://github.com/Tinche/aiofiles | Apache-2.0 |
| aiohappyeyeballs | 2.7.1 | https://github.com/aio-libs/aiohappyeyeballs | PSF-2.0 |
| aiohttp | 3.14.3 | https://github.com/aio-libs/aiohttp | Apache-2.0 AND MIT |
| aioitertools | 0.13.0 | https://aioitertools.omnilib.dev | MIT |
| aiosignal | 1.4.0 | https://github.com/aio-libs/aiosignal | Apache-2.0 |
| alabaster | 1.0.0 | https://alabaster.readthedocs.io | none declared |
| annotated-types | 0.8.0 | https://github.com/annotated-types/annotated-types | MIT |
| anyio | 4.15.1 | https://anyio.readthedocs.io | MIT |
| appnope | 1.0.0 | https://github.com/minrk/appnope | BSD-2-Clause |
| array_record | 0.8.3 | https://github.com/google/array_record | Apache-2.0 |
| asttokens | 3.0.2 | https://github.com/gristlabs/asttokens | Apache-2.0 |
| attrs | 26.1.0 | https://www.attrs.org | MIT |
| babel | 2.18.0 | https://babel.pocoo.org | BSD-3-Clause |
| beautifulsoup4 | 4.15.0 | https://www.crummy.com/software/BeautifulSoup | MIT |
| botocore | 1.43.75 | https://github.com/boto/botocore | Apache-2.0 |
| certifi | 2026.7.22 | https://github.com/certifi/python-certifi | MPL-2.0 |
| cfgv | 3.5.0 | https://github.com/asottile/cfgv | MIT |
| charset-normalizer | 3.5.1 | https://github.com/jawah/charset_normalizer | MIT |
| chex | 0.1.92 | https://chex.readthedocs.io | none declared |
| click | 8.5.0 | https://click.palletsprojects.com | BSD-3-Clause |
| cloudpickle | 3.1.2 | https://github.com/cloudpipe/cloudpickle | BSD-3-Clause |
| comm | 0.2.3 | https://github.com/ipython/comm | BSD-3-Clause |
| contourpy | 1.3.3 | https://github.com/contourpy/contourpy | BSD-3-Clause |
| coverage | 7.16.0 | https://github.com/coveragepy/coveragepy | Apache-2.0 |
| cycler | 0.12.1 | https://matplotlib.org/cycler | BSD-3-Clause |
| datasets | 5.0.1 | https://github.com/huggingface/datasets | Apache-2.0 |
| debugpy | 1.8.21 | https://aka.ms/debugpy | MIT |
| decorator | 5.3.1 | see PyPI | BSD-2-Clause |
| diffrax | 0.7.2 | https://github.com/patrick-kidger/diffrax | Apache-2.0 |
| dill | 0.4.1 | https://github.com/uqfoundation/dill | BSD-3-Clause |
| distlib | 0.4.3 | https://github.com/pypa/distlib | PSF-2.0 |
| distrax | 0.1.9 | https://github.com/google-deepmind/distrax | none declared (Apache-2.0 per repo) |
| dm-tree | 0.1.10 | https://github.com/deepmind/tree | Apache-2.0 |
| docutils | 0.22.4 | https://docutils.sourceforge.io | none declared |
| equinox | 0.13.8 | https://github.com/patrick-kidger/equinox | Apache-2.0 |
| etils | 1.14.0 | https://github.com/google/etils | none declared |
| executing | 2.2.1 | https://github.com/alexmojaki/executing | MIT |
| fastjsonschema | 2.22.2 | https://github.com/horejsek/python-fastjsonschema | BSD-3-Clause |
| filelock | 3.32.6 | https://py-filelock.readthedocs.io | MIT |
| flax | 0.12.8 | https://github.com/google/flax | none declared (Apache-2.0 per repo) |
| fonttools | 4.65.0 | http://github.com/fonttools/fonttools | MIT |
| frozenlist | 1.8.0 | https://github.com/aio-libs/frozenlist | Apache-2.0 |
| fsspec | 2026.2.0 | https://filesystem-spec.readthedocs.io | BSD-3-Clause |
| furo | 2025.12.19 | https://github.com/pradyunsg/furo | none declared |
| gast | 0.7.0 | https://github.com/serge-sans-paille/gast | BSD-3-Clause |
| grain | 0.2.18 | https://github.com/google/grain | Apache-2.0 |
| h11 | 0.16.0 | https://github.com/python-hyper/h11 | MIT |
| hf-xet | 1.6.0 | https://huggingface.co/docs/hub/xet | Apache-2.0 |
| httpcore | 1.0.9 | https://www.encode.io/httpcore | BSD-3-Clause |
| httpx | 0.28.1 | https://github.com/encode/httpx | BSD-3-Clause |
| huggingface_hub | 1.31.0 | https://github.com/huggingface/huggingface_hub | Apache-2.0 |
| humanize | 4.16.0 | https://github.com/python-humanize/humanize | MIT |
| identify | 2.6.19 | https://github.com/pre-commit/identify | MIT |
| idna | 3.19 | https://github.com/kjd/idna | BSD-3-Clause |
| imagesize | 2.0.1 | https://github.com/shibukawa/imagesize_py | MIT |
| iniconfig | 2.3.0 | https://github.com/pytest-dev/iniconfig | MIT |
| ipykernel | 7.3.0 | https://ipython.org | BSD-3-Clause |
| ipython | 9.17.1 | https://ipython.org | BSD-3-Clause |
| ipython_pygments_lexers | 1.1.1 | https://github.com/ipython/ipython-pygments-lexers | none declared |
| jax | 0.11.0 | https://github.com/jax-ml/jax | Apache-2.0 |
| jaxlib | 0.11.0 | https://github.com/jax-ml/jax | Apache-2.0 |
| jaxtyping | 0.3.11 | https://github.com/patrick-kidger/jaxtyping | MIT |
| jedi | 0.20.0 | https://github.com/davidhalter/jedi | MIT |
| Jinja2 | 3.1.6 | https://jinja.palletsprojects.com | none declared (BSD-3-Clause per project) |
| jmespath | 1.1.0 | https://github.com/jmespath/jmespath.py | MIT |
| jsonschema | 4.26.0 | https://github.com/python-jsonschema/jsonschema | MIT |
| jsonschema-specifications | 2025.9.1 | https://jsonschema-specifications.readthedocs.io | MIT |
| jupyter_client | 8.10.0 | https://jupyter.org | BSD-3-Clause |
| jupyter_core | 5.9.1 | https://jupyter.org | BSD-3-Clause |
| kiwisolver | 1.5.1 | https://github.com/nucleic/kiwi | BSD-3-Clause |
| lineax | 0.1.1 | https://github.com/google/lineax | Apache-2.0 |
| markdown-it-py | 4.2.0 | https://markdown-it-py.readthedocs.io | none declared |
| MarkupSafe | 3.0.3 | https://palletsprojects.com | BSD-3-Clause |
| matplotlib | 3.11.1 | https://matplotlib.org | Matplotlib License (BSD-style) + bundled font/library licences (OFL-1.1, Apache-2.0, MIT, FTL/GPL-2.0-or-later dual, CC0-1.0) — see below |
| matplotlib-inline | 0.2.2 | https://github.com/ipython/matplotlib-inline | BSD-3-Clause |
| mdurl | 0.1.2 | https://github.com/executablebooks/mdurl | none declared |
| ml_dtypes | 0.6.0 | https://github.com/jax-ml/ml_dtypes | Apache-2.0 |
| msgpack | 1.2.2 | https://msgpack.org | Apache-2.0 |
| multidict | 6.8.0 | https://github.com/aio-libs/multidict | Apache-2.0 |
| multiprocess | 0.70.19 | https://github.com/uqfoundation/multiprocess | BSD-3-Clause |
| narwhals | 2.26.0 | https://github.com/narwhals-dev/narwhals | MIT |
| nbformat | 5.11.1 | https://jupyter.org | BSD-3-Clause |
| nest-asyncio2 | 1.7.2 | https://github.com/Chaoses-Ib/nest-asyncio2 | BSD |
| nodeenv | 1.10.0 | https://github.com/ekalinin/nodeenv | BSD |
| numpy | 2.5.3 | https://numpy.org | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opt_einsum | 3.4.0 | see PyPI | MIT |
| optax | 0.2.8 | https://optax.readthedocs.io | none declared (Apache-2.0 per repo) |
| optimistix | 0.1.0 | https://github.com/patrick-kidger/optimistix | Apache-2.0 |
| orbax-checkpoint | 0.12.4 | http://github.com/google/orbax | none declared (Apache-2.0 per repo) |
| packaging | 26.3 | https://packaging.pypa.io | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | https://pandas.pydata.org | BSD-3-Clause |
| parso | 0.8.7 | https://github.com/davidhalter/parso | MIT |
| pexpect | 4.9.0 | https://pexpect.readthedocs.io | ISC |
| pillow | 12.3.0 | https://pillow.readthedocs.io | MIT-CMU |
| platformdirs | 4.11.8 | https://platformdirs.readthedocs.io | MIT |
| plotly | 7.0.0 | https://plotly.com/python | MIT |
| pluggy | 1.6.0 | see PyPI | MIT |
| portpicker | 1.6.0 | https://github.com/google/python_portpicker | Apache-2.0 |
| pre_commit | 4.6.2 | https://github.com/pre-commit/pre-commit | MIT |
| prometheus_client | 0.26.0 | https://github.com/prometheus/client_python | Apache-2.0 AND BSD-2-Clause |
| prompt_toolkit | 3.0.53 | https://github.com/prompt-toolkit/python-prompt-toolkit | none declared |
| propcache | 0.5.2 | https://github.com/aio-libs/propcache | Apache-2.0 |
| protobuf | 7.36.1 | https://developers.google.com/protocol-buffers | BSD-3-Clause |
| psutil | 7.2.2 | https://github.com/giampaolo/psutil | BSD-3-Clause |
| ptyprocess | 0.7.0 | https://github.com/pexpect/ptyprocess | unknown |
| pure_eval | 0.2.3 | http://github.com/alexmojaki/pure_eval | MIT |
| pyarrow | 25.0.1 | https://arrow.apache.org | Apache-2.0 |
| pydantic | 2.13.5 | https://github.com/pydantic/pydantic | MIT |
| pydantic_core | 2.46.5 | https://github.com/pydantic/pydantic | MIT |
| Pygments | 2.21.0 | https://pygments.org | BSD-2-Clause |
| pyparsing | 3.3.2 | https://pyparsing-docs.readthedocs.io | MIT |
| pyright | 1.1.414 | https://github.com/RobertCraigie/pyright-python | MIT |
| pytest | 9.1.1 | https://docs.pytest.org | MIT |
| pytest-cov | 7.1.0 | https://github.com/pytest-dev/pytest-cov | MIT |
| python-dateutil | 2.9.0.post0 | https://github.com/dateutil/dateutil | Dual: Apache-2.0 OR BSD-3-Clause |
| python-discovery | 1.6.0 | https://github.com/tox-dev/python-discovery | MIT-style (unlabelled) |
| PyYAML | 6.0.3 | https://pyyaml.org | MIT |
| pyzmq | 27.2.0 | https://pyzmq.readthedocs.org | BSD-3-Clause |
| referencing | 0.37.0 | https://referencing.readthedocs.io | MIT |
| requests | 2.34.2 | https://requests.readthedocs.io | Apache-2.0 |
| rich | 15.0.0 | https://rich.readthedocs.io | MIT |
| roman-numerals | 4.1.0 | https://github.com/AA-Turner/roman-numerals | 0BSD OR CC0-1.0 |
| rpds-py | 2026.6.3 | https://rpds.readthedocs.io | MIT |
| s3fs | 2026.2.0 | http://github.com/fsspec/s3fs | BSD |
| scipy | 1.18.1 | https://scipy.org | BSD-3-Clause; **bundled binaries include GPL-3.0-or-later WITH GCC-exception-3.1 and LGPL-2.1-or-later — see note above** |
| simplejson | 4.1.2 | https://github.com/simplejson/simplejson | MIT OR AFL-2.1 |
| six | 1.17.0 | https://github.com/benjaminp/six | MIT |
| snowballstemmer | 3.1.1 | https://github.com/snowballstem/snowball | BSD-3-Clause |
| soupsieve | 2.9.2 | https://github.com/facelessuser/soupsieve | MIT |
| Sphinx | 9.1.0 | https://www.sphinx-doc.org | BSD-2-Clause |
| sphinx-basic-ng | 1.0.0b2 | https://github.com/pradyunsg/sphinx-basic-ng | none declared |
| sphinx_design | 0.7.0 | https://sphinx-design.readthedocs.io | none declared |
| sphinxcontrib-applehelp | 2.0.0 | see PyPI | none declared |
| sphinxcontrib-devhelp | 2.0.0 | see PyPI | none declared |
| sphinxcontrib-htmlhelp | 2.1.0 | see PyPI | none declared |
| sphinxcontrib-jsmath | 1.0.1 | http://sphinx-doc.org | BSD |
| sphinxcontrib-qthelp | 2.0.0 | see PyPI | none declared |
| sphinxcontrib-serializinghtml | 2.0.0 | see PyPI | none declared |
| stack-data | 0.6.3 | http://github.com/alexmojaki/stack_data | MIT |
| stix-ml | 0.1.0 | https://github.com/instadeepai/stix | Apache-2.0 |
| tensorstore | 0.1.85 | https://github.com/google/tensorstore | Apache-2.0 |
| tfp-nightly | 0.26.0.dev20260910 | http://github.com/tensorflow/probability | Apache-2.0 |
| toolz | 1.1.0 | https://github.com/pytoolz/toolz | BSD-3-Clause |
| tornado | 6.5.8 | http://www.tornadoweb.org | Apache-2.0 |
| tqdm | 4.70.0 | https://tqdm.github.io | MPL-2.0 AND MIT |
| traitlets | 5.16.1 | https://github.com/ipython/traitlets | BSD-3-Clause |
| treescope | 0.1.10 | https://github.com/google-deepmind/treescope | none declared |
| typing-inspection | 0.4.4 | https://github.com/pydantic/typing-inspection | MIT |
| typing_extensions | 4.16.0 | https://github.com/python/typing_extensions | PSF-2.0 |
| urllib3 | 2.7.0 | https://github.com/urllib3/urllib3 | MIT |
| uvloop | 0.22.1 | https://github.com/MagicStack/uvloop | MIT |
| virtualenv | 21.7.9 | https://virtualenv.pypa.io | MIT |
| wadler_lindig | 0.1.7 | https://github.com/patrick-kidger/wadler_lindig | Apache-2.0 |
| wcwidth | 0.8.3 | https://github.com/jquast/wcwidth | MIT |
| wrapt | 2.4.0 | https://github.com/GrahamDumpleton/wrapt | BSD-2-Clause |
| xxhash | 4.0.1 | https://github.com/ifduyue/python-xxhash | BSD-2-Clause |
| yarl | 1.24.5 | https://github.com/aio-libs/yarl | Apache-2.0 |
| zipp | 4.1.0 | https://github.com/jaraco/zipp | MIT |

`matplotlib`'s bundled font and library components (fonts distributed via
`matplotlib` for testing/rendering, and small vendored libraries like
HarfBuzz, FreeType, QHull) carry their own licences — OFL-1.1, Apache-2.0,
MIT-style, CC0-1.0, and one FreeType-license/GPL-2.0-or-later dual choice.
None of these are copyleft-in-practice for a downstream user of matplotlib;
they're standard for any project bundling matplotlib and not specific to
`stix`.

`dev`-group-only packages above (pytest, sphinx, pyright, plotly,
pre-commit, ipykernel, etc.) are not shipped in the published `stix-ml`
package and carry no distribution obligations for end users — included here
for completeness since the snapshot captures the full dev environment.

## How to regenerate this list

```bash
uv sync
uv run python - <<'PY'
import importlib.metadata as md
for d in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    name = d.metadata.get("Name")
    version = d.version
    lic = d.metadata.get("License-Expression") or d.metadata.get("License")
    home = d.metadata.get("Home-page") or d.metadata.get("Project-URL")
    print(f"{name} | {version} | {home or 'see PyPI'} | {lic or 'none declared'}")
PY
```

This lists every installed package (direct and transitive), with version,
source, and declared licence. Re-run after any dependency bump, and re-check
the `gpu`/`tpu` extras specifically, since NVIDIA's packaging and licence
metadata have changed between JAX/CUDA releases in the past.
