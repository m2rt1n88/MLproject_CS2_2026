from contextlib import closing
import argparse
import json
from pathlib import Path

from .store import ARCHIVE, BASE, CACHE_DIRS, STORE, import_cache, open_store, page_path, seed, status


def main():
    p = argparse.ArgumentParser(description='Resume HLTV detail backfill from csgopred and local HTML; never sends requests.')
    p.add_argument('--store', type=Path, default=STORE)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('seed', help='reuse read-only csgopred archive IDs')
    s.add_argument('--since', default='2024-01-21')
    s.add_argument('--until')
    s.add_argument('--id', type=int, action='append', default=[])
    plan = sub.add_parser('plan', help='list exact fields/URLs still needed')
    plan.add_argument('--limit', type=int, default=25)
    plan.add_argument('--id', type=int, action='append', default=[])
    sub.add_parser('status')
    gaps = sub.add_parser('gaps', help='prioritize verified dataset candidates with incomplete local box scores')
    gaps.add_argument('--limit', type=int, default=25)
    i = sub.add_parser('import-cache', help='read ID.html or the original csgopred cache filenames')
    i.add_argument('--folder', type=Path, action='append')
    i.add_argument('--id', type=int, action='append', default=[])
    i.add_argument('--limit', type=int, default=100)
    i.add_argument('--resume', action='store_true', help='resume after checking previously blocked HTML')
    i.add_argument('--reparse-invalid', action='store_true', help='recheck rejected cache files after a parser fix')
    e = sub.add_parser('export', help='export validated clean_v3 history events (not new labels)')
    e.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    with closing(open_store(args.store)) as db:
        if args.action == 'seed':
            result = seed(db, args.archive, since=args.since, until=args.until, ids=args.id)
        elif args.action == 'status':
            result = status(db)
        elif args.action == 'gaps':
            if args.limit < 1:
                p.error('--limit must be positive')
            from .history import gap_inventory
            from ..features import load_source

            matches, maps, players, _, _ = load_source()
            result = gap_inventory(db, args.archive, matches, maps, players, limit=args.limit)
        elif args.action == 'plan':
            if args.limit < 1:
                p.error('--limit must be positive')
            rows = db.execute('''SELECT * FROM progress
                WHERE scores!='page' OR veto!='complete' OR player_maps!=expected_maps ORDER BY day,id''')
            wanted = set(args.id)
            result = []
            for r in rows:
                if wanted and r['id'] not in wanted:
                    continue
                saved = [s[0] for s in db.execute('SELECT name FROM accepted_players WHERE id=? ORDER BY name', (r['id'],))]
                last = db.execute('SELECT error FROM observations WHERE id=? ORDER BY rowid DESC LIMIT 1', (r['id'],)).fetchone()
                result.append(dict(r) | {'url': BASE + page_path(r), 'player_maps_saved': saved,
                                         'last_error': last[0] if last else None})
                if len(result) == args.limit:
                    break
        elif args.action == 'import-cache':
            folders = args.folder if args.folder is not None else CACHE_DIRS
            result = import_cache(db, folders, ids=args.id, limit=args.limit, resume=args.resume,
                                  reparse_invalid=args.reparse_invalid)
        else:
            from .history import build_patch, read_patch
            from ..features import excluded_map_keys, load_source

            matches, maps, players, digests, _ = load_source()
            excluded = excluded_map_keys()
            patch = build_patch(db, args.archive, matches, maps, players, digests, excluded)
            # Do not overwrite an experiment or dataset by mistake. Validate before creating the final file.
            from tempfile import TemporaryDirectory

            raw = json.dumps(patch, ensure_ascii=False, indent=2) + '\n'
            with TemporaryDirectory() as tmp:
                candidate = Path(tmp) / 'history.json'
                candidate.write_text(raw, encoding='utf-8')
                read_patch(candidate, matches, maps, players, digests, excluded)
            with args.out.open('x', encoding='utf-8') as f:
                f.write(raw)
            result = {'out': str(args.out.resolve()), **patch['report']}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
