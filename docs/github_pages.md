# GitHub Pages deployment

The public research hub is published at
https://hlin863.github.io/ewma_covariate_shift_demo/.

The `Deploy research hub to GitHub Pages` workflow exports the existing Flask
templates and committed results on each push to `main`. It can also be started
from the repository's Actions tab. In Settings → Pages, the publishing source
must be **GitHub Actions**. This is a one-time repository setting.

## What is published

- The research home, process explanations, dataset structures and result pages.
- Existing JavaScript charts and locally selectable chart controls.
- Saved CSV/JSON/figure downloads referenced by those pages.
- Pre-rendered selections for GET forms, including synthetic scenarios and
  dataset subsets. These appear as links on the published site.
- The latest committed Diethe experiment, explicitly identified by its path.
- Saved test reports, when present, with automatic test execution disabled.

Each page identifies the source commit and build time. Build time is not the
experiment date. The exporter does not run experiments, pytest or Ollama.
Missing EEG recordings and outputs remain visibly unavailable. The paper links
open the PDFs at the exact source commit on GitHub.

GitHub Pages cannot host the Flask server. New experiments, arbitrary server
queries, raw EEG processing, test reruns and local RAG/Ollama require `python
app.py`. Their server controls are replaced by local-use instructions in the
export only. The Python application retains its local behaviour. The deployment
change also fixes a Jinja dictionary-key lookup on the adaptive-learning page
that otherwise raised a rendering error.

## Build locally

Use a clean checkout so the snapshot reflects committed evidence:

```bash
python -m pip install -r requirements-pages.txt
python -m scripts.build_github_pages --base-path /ewma_covariate_shift_demo
```

The exporter writes `_site/`, which is ignored by Git. It requires a fresh output
directory to prevent stale files from entering a deployment. Use `--output` to
choose another directory. For local serving at the root, build with
`--base-path '' --output /tmp/ewma-pages-preview`, then run
`python -m http.server --directory /tmp/ewma-pages-preview`.

The build crawls the rendered navigation and form selections, checks response
status, copies only Git-tracked referenced assets, validates every internal
HTML link and asset under the repository prefix, and writes
`build-manifest.json` with the source commit and route map. A failed build blocks
deployment. The deployment job uses GitHub's Pages artifact and OIDC workflow,
with read-only repository access and Pages write permission for deployment.
