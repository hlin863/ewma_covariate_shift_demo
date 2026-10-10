"""Export the existing Flask UI and committed evidence for GitHub Pages.

No experiments, test suites or model requests are run by this exporter. Server
forms become links to pre-rendered selections; browser charts remain interactive.
Only Git-tracked static assets and output downloads can enter the published site.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit

from flask import render_template

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/hlin863/ewma_covariate_shift_demo'


def compatible_diethe_snapshot(result: dict) -> bool:
    """Current six-policy template requires evidence-value decisions and metadata.

    Legacy five-policy snapshots remain valid historical results but must not be
    passed to a template expecting fields they did not record.
    """
    if not isinstance(result, dict):
        return False
    runs = result.get('runs')
    metadata = result.get('metadata')
    summary = result.get('summary')
    if not isinstance(runs, dict) or not isinstance(metadata, dict) or not isinstance(summary, list):
        return False
    evidence = runs.get('evidence_value')
    config = metadata.get('config')
    if not isinstance(evidence, dict) or not isinstance(evidence.get('evidence_decisions'), list):
        return False
    if not isinstance(config, dict):
        return False
    required_config = {'interval', 'performance_window', 'update_scope'}
    required_row = {'policy_family', 'trigger'}
    return required_config.issubset(config) and all(
        isinstance(row, dict) and required_row.issubset(row) for row in summary
    )


def canonical(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return parts.path + ('?' + query if query else '')


def page_path(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.strip('/')
    if parts.query:
        path += '/selection-' + sha256(parts.query.encode()).hexdigest()[:12]
    return (path + '/' if path else '') + 'index.html'


def export(base_path: str, destination: Path) -> dict:
    from bs4 import BeautifulSoup
    from src.web import app

    if destination.exists():
        raise ValueError(f'{destination} already exists; use a fresh output directory.')
    destination.mkdir(parents=True)
    app.config.update(TESTING=True, TEST_RESULTS_AUTO_RUN=False)
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT, text=True).split('\0'))
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    base = '/' + base_path.strip('/') if base_path.strip('/') else ''
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    queue = deque(['/'])
    pages = {}

    def link(url: str, current: str) -> str:
        if not url or url.startswith('#'):
            return url
        absolute = urlsplit(urljoin('https://export.invalid' + current, url))
        if absolute.netloc != 'export.invalid' or absolute.scheme != 'https':
            return url
        path = unquote(absolute.path).lstrip('/')
        if absolute.path.startswith('/papers/'):
            return REPOSITORY + '/blob/' + commit + '/' + quote(path)
        if absolute.path.startswith(('/static/', '/outputs/')):
            if path not in tracked or not (ROOT / path).is_file():
                raise ValueError(f'Untracked or missing published asset: {path}')
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / path).read_bytes())
            return base + '/' + quote(path) + ('#' + absolute.fragment if absolute.fragment else '')
        if absolute.path.startswith('/api/'):
            raise ValueError(f'Live API cannot be exported: {url}')
        normalized = canonical(absolute.path + ('?' + absolute.query if absolute.query else ''))
        if normalized not in pages:
            queue.append(normalized)
        return base + '/' + page_path(normalized) + ('#' + absolute.fragment if absolute.fragment else '')

    client = app.test_client()
    while queue:
        url = queue.popleft()
        if url in pages:
            continue
        # Mark before rendering so self-links cannot create an unbounded crawl.
        pages[url] = page_path(url)
        if len(pages) > 2000:
            raise ValueError('Unexpectedly large route graph')
        saved_run = None
        if url == '/results/diethe':
            runs = sorted(p for p in tracked if p.startswith('outputs/diethe/') and p.endswith('/experiment.json'))
            if runs:
                candidate = runs[-1]
                result = json.loads((ROOT / candidate).read_text())
                if compatible_diethe_snapshot(result):
                    saved_run = candidate
                    with app.test_request_context(url):
                        html = render_template('diethe.html', values=result['metadata']['config'], result=result, error=None)
                else:
                    # Preserve legacy evidence on disk; do not fabricate a sixth
                    # policy or pass old fields to a newer template.
                    html = client.get(url).get_data(as_text=True)
                    with app.test_request_context(url):
                        legacy_note = (
                            'Historical five-policy Diethe evidence is retained at '
                            + candidate + '. It predates the evidence-value policy, '
                            'so it is not displayed as a six-policy comparison. '
                            'Use local Flask to run the current experiment.'
                        )
            else:
                html = client.get(url).get_data(as_text=True)
        elif url == '/support':
            # Do not index local files or connect to Ollama during a public build.
            with app.test_request_context(url):
                html = render_template('support.html', query='', result=None,
                                       status=None, page_error='Run python app.py locally to use repository retrieval and Ollama.')
        else:
            response = client.get(url)
            if response.status_code != 200:
                raise ValueError(f'{url}: HTTP {response.status_code}')
            html = response.get_data(as_text=True)
        soup = BeautifulSoup(html, 'html.parser')
        banner = soup.new_tag('aside', attrs={'class': 'pages-snapshot', 'aria-label': 'Published snapshot'})
        banner.string = f'Published research snapshot · {stamp} · {commit[:7]}. Charts and saved evidence are available here. New runs and local data require python app.py.'
        soup.body.insert(0, banner)
        style = soup.new_tag('style')
        style.string = '.pages-snapshot{padding:12px 24px;background:#132a39;color:#eef7ff;font:14px/1.5 system-ui;border-bottom:1px solid #467084}.pages-options{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.pages-options a{padding:6px 10px;border:1px solid #69808f;border-radius:6px}.pages-local{padding:12px;border:1px solid #69808f;border-radius:6px}'
        soup.head.append(style)
        if saved_run:
            note = soup.new_tag('p', attrs={'class': 'pages-local'})
            note.string = 'Saved experiment: ' + saved_run + '. These metrics were loaded from the committed run; the website build did not rerun it.'
            soup.find('header').append(note)
        elif url == '/results/diethe' and runs and not compatible_diethe_snapshot(result):
            note = soup.new_tag('p', attrs={'class': 'pages-local'})
            note.string = legacy_note
            soup.find('header').append(note)
        for form in list(soup.find_all('form')):
            replacement = soup.new_tag('div', attrs={'class': 'pages-options'})
            if form.get('method', 'get').lower() == 'post':
                replacement['class'] = 'pages-local'
                replacement.string = 'Run this feature locally: python app.py. GitHub Pages serves saved results and cannot run Python or Ollama.'
            else:
                fields = []
                for field in form.select('select[name], input[name]'):
                    if field.name == 'select':
                        options = [(opt.get('value', opt.get_text(strip=True)), opt.get_text(strip=True)) for opt in field.find_all('option')]
                    else:
                        options = [(field.get('value', ''), field.get('value', ''))]
                    fields.append((field['name'], options))
                if not fields:
                    raise ValueError(f'GET form without exportable controls: {url}')
                # Some live filters (e.g. Park: source x subject x session)
                # have hundreds of combinations. A static snapshot cannot
                # expose them all or necessarily load their local datasets.
                # Keep the already-rendered default and explain the boundary.
                combination_count = 1
                for _, options in fields:
                    combination_count *= len(options)
                if combination_count > 100:
                    replacement['class'] = 'pages-local'
                    replacement.string = (
                        f'This interactive filter has {combination_count} possible '
                        'selections. The published snapshot shows the default '
                        'view; run python app.py locally to use all filters.'
                    )
                else:
                    action = urlsplit(urljoin(url, form.get('action', url))).path
                    for values in product(*(options for _, options in fields)):
                        query = urlencode([(field[0], value[0]) for field, value in zip(fields, values)])
                        a = soup.new_tag('a', href=action + '?' + query)
                        a.string = ' · '.join(value[1] for value in values)
                        replacement.append(a)
            form.replace_with(replacement)
        # Local test controls must not suggest that a static site can rerun pytest.
        for a in list(soup.select('a[href]')):
            if 'refresh=1' in a['href']:
                a.replace_with('Run tests locally with python -m pytest -q.')
        for script in soup.find_all('script'):
            if script.string and 'async function pollResults()' in script.string:
                start = script.string.index('      async function pollResults()')
                end = script.string.index('      window.setInterval(pollResults, refreshMs);') + len('      window.setInterval(pollResults, refreshMs);')
                script.string = script.string[:start] + script.string[end:]
        for tag in soup.find_all(True):
            for attr in ('href', 'src'):
                if tag.has_attr(attr):
                    tag[attr] = link(tag[attr], url)
        target = destination / pages[url]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(soup).replace(str(ROOT), '.'), encoding='utf-8')

    # Fail the build on a broken internal page or asset URL, including subpath errors.
    checked = 0
    for path in destination.rglob('*.html'):
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        for tag in soup.find_all(True):
            for attr in ('href', 'src'):
                value = tag.get(attr, '')
                if not value.startswith('/') or value.startswith('//'):
                    continue
                if not value.startswith(base + '/'):
                    raise ValueError(f'Link escaped project base: {value}')
                relative = unquote(urlsplit(value).path[len(base) + 1:])
                if not (destination / relative).is_file():
                    raise ValueError(f'Broken export link in {path}: {value}')
                checked += 1
    manifest = {'source_commit': commit, 'built_at': stamp, 'base_path': base,
                'page_count': len(pages), 'checked_links': checked, 'routes': pages}
    (destination / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / '.nojekyll').touch()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-path', default='/ewma_covariate_shift_demo')
    parser.add_argument('--output', type=Path, default=ROOT / '_site')
    args = parser.parse_args()
    manifest = export(args.base_path, args.output.resolve())
    print(f"Exported {manifest['page_count']} pages; validated {manifest['checked_links']} internal links.")
