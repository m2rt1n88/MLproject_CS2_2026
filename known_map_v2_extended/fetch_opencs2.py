"""Cache the pinned public OpenCS2 demo tables (not the raw HLTV demos)."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from time import sleep
from itertools import islice
from urllib.request import urlopen

REPO = 'blanchon/opencs2_dataset_demo'
BASE = 'https://huggingface.co'
OUT = Path(__file__).resolve().parent / 'runs/opencs2_public'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / 'index.json'
    if index.exists():
        version = json.loads(index.read_text(encoding='utf8'))
    else:
        with urlopen(f'{BASE}/api/datasets/{REPO}', timeout=25) as response:
            info = json.load(response)
        version = dict(repo=REPO, revision=info['sha'], files=sorted(
            f['rfilename'] for f in info['siblings'] if f['rfilename'].endswith('.parquet')))
        index.write_text(json.dumps(version, indent=2), encoding='utf8')
    sha = version['revision']
    if (version['repo'] != REPO or not re.fullmatch('[0-9a-f]{40}', sha)
            or not 1 <= len(version['files']) <= 180
            or not all(re.fullmatch('data/[a-zA-Z0-9_.-]+[.]parquet', f) for f in version['files'])):
        raise ValueError('Unexpected published file list')

    def fetch(path):
        output = OUT / Path(path).name
        if output.exists():
            return output.stat().st_size
        sleep(.3)
        with urlopen(f'{BASE}/datasets/{REPO}/resolve/{sha}/{path}', timeout=25) as response:
            data = response.read(2_000_001)
        if not 8 <= len(data) <= 2_000_000 or not data.startswith(b'PAR1') or not data.endswith(b'PAR1'):
            raise ValueError(f'Invalid public parquet shard {path}')
        with output.open('xb') as file:
            file.write(data)
        return len(data)

    sizes = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = iter(version['files'])
        while batch := list(islice(files, 3)):
            sizes.extend(pool.map(fetch, batch))
            if len(sizes) % 21 == 0:
                print(len(sizes), '/', len(version['files']), flush=True)
    print(dict(revision=sha, files=len(sizes), bytes=sum(sizes)))


if __name__ == '__main__':
    main()
