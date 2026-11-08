"""One JSON file per article in a dedicated GitHub repository.

Only use server-side credentials scoped to the news repository.
"""
import base64
import json
import re
import uuid
from datetime import date
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class StorageError(RuntimeError):
    pass


def api(config, path, method='GET', payload=None):
    repo = config['repo']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('NEWS_GITHUB_REPO must be owner/repository.')
    req = Request('https://api.github.com/repos/' + repo + path,
                  data=json.dumps(payload).encode() if payload is not None else None,
                  headers={'Authorization': 'Bearer ' + config['token'],
                           'Accept': 'application/vnd.github+json',
                           'Content-Type': 'application/json',
                           'User-Agent': 'FinancialResearchNews'}, method=method)
    try:
        with urlopen(req, timeout=25) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise StorageError('GitHub file or repository not found. Check repository name, branch and token access.') from None
        if exc.code in (409, 422):
            raise StorageError('GitHub rejected a conflicting update. Refresh and try again; check branch protection rules.') from None
        if exc.code in (401, 403, 429):
            raise StorageError('GitHub access denied or rate limit reached. Check token permissions/expiry, or retry later.') from None
        raise StorageError(f'GitHub storage request failed (HTTP {exc.code}).') from None
    except (URLError, TimeoutError):
        raise StorageError('Could not reach GitHub storage. Try again later.') from None


def content_path(config, filename=''):
    return '/contents/news' + ('/' + quote(filename, safe='') if filename else '')


def branch_query(config):
    return '?ref=' + quote(config['branch'], safe='')


@lru_cache(maxsize=1500)
def read_blob(repo, token, branch, filename, sha):
    config = dict(repo=repo, token=token, branch=branch)
    # Read immutable blob SHA so cached data always matches the listing revision.
    result = api(config, '/git/blobs/' + quote(sha, safe=''))
    if result.get('encoding') != 'base64':
        raise StorageError('Unexpected GitHub article encoding.')
    data = json.loads(base64.b64decode(result['content']).decode('utf-8'))
    required = ('headline', 'published_date', 'body', 'analysis', 'source_url')
    if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in required):
        raise StorageError('An article has an invalid format. Check the news repository.')
    return {**data, 'id': filename, '_sha': sha}


def list_posts(config):
    entries = api(config, content_path(config) + branch_query(config))
    if not isinstance(entries, list):
        raise StorageError('Expected a news folder in the repository.')
    if len(entries) >= 1000:
        raise StorageError('News folder reached its listing limit. Archive by year before adding more posts.')
    files = [e for e in entries if e.get('type') == 'file' and e['name'].endswith('.json')]
    def read(entry):
        return read_blob(config['repo'], config['token'], config['branch'], entry['name'], entry['sha'])
    with ThreadPoolExecutor(max_workers=5) as pool:
        posts = list(pool.map(read, files))
    return sorted(posts, key=lambda p: (p['published_date'], p['id']), reverse=True)


def save_post(config, headline, published_date, body, analysis='', source_url='', post_id=None, expected_sha=None):
    headline, body, analysis, source_url = (v.strip() for v in (headline, body, analysis, source_url))
    if not headline or not body:
        raise ValueError('Headline and news text are required.')
    if len(headline) > 250 or len(body) > 50000 or len(analysis) > 20000:
        raise ValueError('Article exceeds the field length limits.')
    day = date.fromisoformat(str(published_date))
    if day > date.today():
        raise ValueError('Choose today or an earlier date.')
    if source_url:
        parsed = urlparse(source_url)
        if parsed.scheme not in ('https', 'http') or not parsed.netloc:
            raise ValueError('Source link must be a valid http:// or https:// URL.')
    filename = post_id or f'{day.isoformat()}-{uuid.uuid4().hex}.json'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', filename):
        raise ValueError('Invalid article identifier.')
    if post_id and not expected_sha:
        raise ValueError('Refresh the article before editing.')
    article = dict(headline=headline, published_date=day.isoformat(), body=body,
                   analysis=analysis, source_url=source_url)
    payload = dict(message=('Update ' if post_id else 'Publish ') + filename,
                   branch=config['branch'],
                   content=base64.b64encode(json.dumps(article, ensure_ascii=False, indent=2).encode()).decode())
    if expected_sha:
        payload['sha'] = expected_sha
    api(config, content_path(config, filename), 'PUT', payload)
    return filename


def delete_post(config, post_id, expected_sha=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', post_id) or not expected_sha:
        raise ValueError('Refresh the article before deleting.')
    api(config, content_path(config, post_id), 'DELETE',
        dict(message='Delete ' + post_id, sha=expected_sha, branch=config['branch']))
