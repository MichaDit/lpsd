#!/usr/bin/env python3
"""Read-only audit of local trailers and, optionally, GitHub's rendered authors.

Run from a full Git checkout. The optional web check needs no credentials and
fails visibly if GitHub's page format changes; a trailer is not treated as
proof that GitHub linked the coauthor account.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen


UPSTREAM = '2fd15da6930d19f5978f7e37b7b0785ce560f7d3'
COAUTHOR = 'Codex <267193182+codex@users.noreply.github.com>'


class _EmbeddedJSON(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.current = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('type') == 'application/json':
            self.active = True
            self.current = []

    def handle_data(self, data):
        if self.active:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.blocks.append(''.join(self.current))
            self.active = False


def _git(*args, input_text=None):
    return subprocess.check_output(['git', *args], input=input_text, text=True)


def _web_authors(repository, sha):
    url = f'https://github.com/{repository}/commit/{sha}'
    try:
        request = Request(url, headers={'User-Agent': 'lpsd-coauthor-audit/1'})
        with urlopen(request, timeout=30) as response:
            data = response.read(8 * 1024**2 + 1)
        if len(data) > 8 * 1024**2:
            raise ValueError('Commit page exceeds the 8 MiB audit limit.')
        parser = _EmbeddedJSON()
        parser.feed(data.decode('utf-8'))
        for block in parser.blocks:
            obj = json.loads(block)
            if not isinstance(obj, dict):
                continue
            payload = obj.get('payload', {})
            if not isinstance(payload, dict):
                continue
            route = payload.get('commitRoute', {})
            commit = route.get('commit', {}) if isinstance(route, dict) else {}
            if commit.get('oid') == sha:
                authors = commit.get('authors')
                if not isinstance(authors, list):
                    raise ValueError('Rendered commit has no author list.')
                # Retain only public attribution evidence, not session/page data.
                logins = [author.get('login') for author in authors]
                return {'url': url, 'rendered_logins': logins,
                        'codex_visible': 'codex' in logins, 'error': None}
        raise ValueError('Expected GitHub commit JSON was not found; inspect the page.')
    except Exception as exc:
        return {'url': url, 'rendered_logins': None, 'codex_visible': False,
                'error': f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=UPSTREAM, help='Excluded upstream commit.')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--github', action='store_true', help='Also inspect public GitHub pages.')
    parser.add_argument('--repository', default='MichaDit/lpsd')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
        parser.error('--repository must be owner/name.')
    base = _git('rev-parse', '--verify', f'{args.base}^{{commit}}').strip()
    head = _git('rev-parse', '--verify', f'{args.head}^{{commit}}').strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', base, head], check=True)
    shas = _git('rev-list', '--reverse', f'{base}..{head}').splitlines()
    if not shas:
        parser.error('The selected range contains no commits.')
    rows = []
    for sha in shas:
        message = _git('show', '-s', '--format=%B', sha)
        trailers = _git('interpret-trailers', '--parse', input_text=message)
        rows.append({'sha': sha, 'subject': message.splitlines()[0],
                     'linked_trailer': f'Co-authored-by: {COAUTHOR}' in trailers.splitlines()})
    if args.github:
        with ThreadPoolExecutor(max_workers=4) as pool:
            checks = pool.map(lambda sha: _web_authors(args.repository, sha), shas)
            for row, check in zip(rows, checks):
                row.update(check)
    ok = all(row['linked_trailer'] and (not args.github or row['codex_visible'])
             for row in rows)
    report = {'schema_version': 1, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'repository': args.repository, 'base': base, 'head': head,
              'required_coauthor': COAUTHOR, 'github_checked': args.github,
              'commit_count': len(rows), 'passed': ok, 'commits': rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'commits'}, indent=2))
    for row in rows:
        if not row['linked_trailer'] or (args.github and not row['codex_visible']):
            print(json.dumps(row))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
