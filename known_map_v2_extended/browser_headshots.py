"""Strict HLTV mapstats K(hs) import from browser-saved pages (no network access).

Only saved *mapstats* pages carry K(hs); saved match pages show K-D/ADR/KAST/rating but no headshots,
so a series without saved mapstats pages contributes nothing here. Identity reuses `offline_mapstats`
(series link + two distinct mapstats fingerprints no later than the day); prior links only add conflicts.

Outputs (one run directory, never overwritten):
- `headshot_patch.json`: metric_source `hltv_verified_map_headshots`, metrics `{headshot_kills}` only;
- `metric_history.json`: base history + patch. A key that already has Rating 3.0 becomes one
  `hltv_verified_map_multi_source` row whose `metric_provenance` keeps both full source rows;
- `capture.json`, `hltv_bulk_derived_manifest.json`, `import_report.json`.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from known_map_recent.hltv_backfill.parse import Blocked, InvalidPage
from known_map_v2_extended import offline_mapstats as om
from known_map_v2_extended.metric_history import (CORE, HS_SCHEMA, HS_SOURCE, KEY, MULTI_SOURCE, RATING_SOURCE,
                                                  VERSION, provenance_digest, read_metrics)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BULK, VERIFIED = ROOT / 'temp/hltv_bulk', ROOT / 'temp/hltv_verified'
PILOT = HERE / 'data/browser_pilot_v1/dataset'
BASE, PATCH, OPENCS2 = PILOT / 'metric_history.json', PILOT / 'hltv_history_patch.json', PILOT / 'headshot_history.json'
OUT = ROOT / 'temp/browser_headshots/run_v1'
REPORTS = ROOT / 'reports/browser_headshots'
FILE = re.compile(r'(match|mapstats)_([1-9]\d*)\.html')
HREF = re.compile(r'href="(/stats/matches/mapstatsid/([1-9]\d*)/[^"?#]+)"')
KHS_MARKUP = re.compile(r'st-kills|K\s*\(hs\)')
DERIVED_METHOD = ('pages saved by the user Chrome Browser MCP session into temp/hltv_bulk without a manifest; '
                  'manifest derived offline by browser_headshots (match URL = page canonical, '
                  'mapstats URL = link on that match page, repeated on the mapstats page)')
TIME_EVIDENCE = 'filesystem LastWriteTimeUtc of the saved file; not historical publication time'


def mtime_utc(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat()


def derive_manifest(folder):
    """Manifest for match pages that link at least one saved mapstats page, plus those mapstats pages."""
    folder = Path(folder)
    saved = {int(m[2]): p for p in sorted(folder.glob('mapstats_*.html')) for m in [FILE.fullmatch(p.name)] if m}
    files, owner = [], {}
    for p in sorted(folder.glob('match_*.html')):
        m = FILE.fullmatch(p.name)
        if not m:
            continue
        html = p.read_text(encoding='utf-8')
        links = {}
        for x in HREF.finditer(html):
            if links.setdefault(int(x[2]), x[1]) != x[1]:
                raise InvalidPage(f'{p.name} links mapstats {x[2]} under two paths')
        wanted = sorted(set(links) & set(saved))
        if not wanted:
            continue
        canon = om.soup_of(html).select_one('link[rel="canonical"]')
        url = canon.get('href', '') if canon else ''
        ident = om.MATCH_URL.fullmatch(url)
        if not ident or ident[1] != m[2]:
            raise InvalidPage(f'{p.name}: canonical URL missing or carries another match ID')
        files.append(dict(file=p.name, url=url, captured_utc=mtime_utc(p), sha256=om.sha256(p)))
        for mid in wanted:
            if mid in owner:
                raise InvalidPage(f'mapstats {mid} linked by {owner[mid]} and {p.name}')
            owner[mid] = p.name
            if f'href="{links[mid]}"' not in saved[mid].read_text(encoding='utf-8'):
                raise InvalidPage(f'mapstats_{mid}.html does not repeat its URL {links[mid]}')
            files.append(dict(file=saved[mid].name, url='https://www.hltv.org' + links[mid],
                              captured_utc=mtime_utc(saved[mid]), sha256=om.sha256(saved[mid])))
    return dict(method=DERIVED_METHOD, capture_time_evidence=TIME_EVIDENCE, files=files), sorted(set(saved) - set(owner))


def scan_match_khs(folders):
    """Read-only check that saved match pages carry no K(hs) markup (so they cannot supply headshots)."""
    pages = [p for f in folders for p in sorted(Path(f).glob('match_*.html'))]
    with_khs = [p.name for p in pages if KHS_MARKUP.search(p.read_text(encoding='utf-8', errors='replace'))]
    return dict(match_pages_scanned=len(pages), match_pages_with_khs_markup=len(with_khs), examples=with_khs[:10])
def collect_pages(stage, manifest_folders, derive_folders):
    """Pages from folders with their own capture_manifest.json plus folders whose manifest is derived here."""
    pages, sources, orphans = {}, [], {}
    jobs = [(Path(f), Path(f) / 'capture_manifest.json', 'capture_manifest.json') for f in manifest_folders]
    for i, f in enumerate(map(Path, derive_folders)):
        derived, orphans[str(f)] = derive_manifest(f)
        name = f'derived_manifest_{i}_{f.name}.json'
        om.write_json(stage / name, derived)
        jobs.append((f, stage / name, name))
    for folder, manifest, name in jobs:
        meta, found = om.read_manifest(folder, manifest)
        for k, e in found.items():
            if k in pages and pages[k]['sha256'] != e['sha256']:
                raise InvalidPage(f'two different saved copies of {k[0]}_{k[1]}.html')
            pages.setdefault(k, e)
        sources.append(dict(folder=rel(folder), manifest=name, manifest_sha256=om.sha256(manifest),
                            pages=len(found), **meta))
    return pages, sources, orphans


def rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def headshot_row(series, mid, flip, m, r, side, local_id, links, imported_utc, pages):
    """Own row per value; rating and other page columns are deliberately not copied here."""
    return dict(match_id=mid, map_name=m['map_name'], day=series['day'], side=side, player_id=local_id,
                kills=r['kills'], deaths=r['deaths'], adr=r['adr'], kast=r['kast'],
                scores=om.oriented(m['scores'], flip), rounds=sum(m['scores']), hltv_id=series['match_id'],
                mapstats_id=m['mapstats_id'], hltv_player_id=r['hltv_player_id'],
                identity_proven_by=list(links), identity_as_of=max(links.values()),
                metrics={'headshot_kills': r['headshot_kills']}, metric_source=HS_SOURCE, metric_schema=HS_SCHEMA,
                sha256=m['sha256'], source=m['source_url'], source_url=m['source_url'],
                source_file=rel(pages['mapstats', m['mapstats_id']]['path']),
                match_source_url=series['source_url'], match_sha256=series['sha256'],
                match_source_file=rel(pages['match', series['match_id']]['path']),
                captured_utc=m['captured_utc'], match_captured_utc=series['captured_utc'],
                imported_utc=imported_utc, match_page_box_checked=m['match_page_box_checked'])


def multi_row(old, h):
    prov = {'rating': old, 'headshot_kills': h}
    return dict({c: old[c] for c in CORE}, scores=old.get('scores'), hltv_id=old.get('hltv_id'),
                mapstats_id=old.get('mapstats_id'), hltv_player_id=old['hltv_player_id'],
                metrics={'rating': old['metrics']['rating'], 'headshot_kills': h['metrics']['headshot_kills']},
                rating_version='3.0', metric_source=MULTI_SOURCE, source='metric_provenance',
                sha256=provenance_digest(prov), imported_utc=h['imported_utc'], metric_provenance=prov)


def merge_headshots(base_rows, rows, evidence=None):
    """Add K(hs) by key; an existing Rating 3.0 row becomes one multi-source row. Never overwrites a value.

    An identical K(hs)/core/identity re-import from a page with a different SHA keeps the old source and,
    if `evidence` is a list, appends the new capture there; any value/core/identity difference raises.
    """
    out, where = list(base_rows), {}
    for i, r in enumerate(out):
        k = tuple(r[c] for c in KEY)
        if k in where:
            raise ValueError(f'Duplicate key in base metric history for {k}')
        where[k] = i
    counts, skipped = Counter(), []
    for h in rows:
        k = tuple(h[c] for c in KEY)
        i = where.get(k)
        old = None if i is None else out[i]
        src = None if old is None else old['metric_source']
        if old is None:
            where[k] = len(out)
            out.append(h)
            counts['new_headshot_only_rows'] += 1
        elif src in (HS_SOURCE, MULTI_SOURCE):
            prev = old if src == HS_SOURCE else old['metric_provenance']['headshot_kills']
            if (prev['metrics'] != h['metrics'] or any(prev[c] != h[c] for c in CORE)
                    or prev['hltv_player_id'] != h['hltv_player_id']):
                raise ValueError(f'Conflicting headshot row for {k}')
            if prev['sha256'] == h['sha256']:
                counts['duplicate_existing_headshot_rows'] += 1
            else:
                # Same value from a re-saved page (ads/markup drift): keep the old source, retain the new one.
                counts['duplicate_content_new_capture'] += 1
                if evidence is not None:
                    evidence.append(dict(key=list(k), kept_sha256=prev['sha256'], kept_source=prev['source'],
                                         new_capture=h))
        elif (src == RATING_SOURCE and set(old['metrics']) == {'rating'}
              and all(old[c] == h[c] for c in CORE) and old.get('hltv_player_id') == h['hltv_player_id']):
            out[i] = multi_row(old, h)
            counts['rating_rows_combined_with_headshots'] += 1
        else:
            reason = 'base_core_or_identity_differs' if src == RATING_SOURCE else f'base_source_{src}_not_combinable'
            skipped.append(dict(key=list(k), reason=reason))
            counts['not_merged_' + reason] += 1
    return out, dict(counts), skipped


def opencs2_check(rows, opencs2, players, hashes):
    """Demo-derived OpenCS2 headshots are independent evidence; a disagreeing key is dropped, not resolved."""
    from known_map_v2_extended.opencs2_headshots import read_history

    demo = {} if opencs2 is None else read_history(opencs2, players, hashes)
    keep, report = [], Counter()
    conflicts = []
    for h in rows:
        k = tuple(h[c] for c in KEY)
        if k not in demo:
            keep.append(h)
        elif demo[k] == h['metrics']['headshot_kills']:
            report['agree_with_opencs2'] += 1
            keep.append(h)
        else:
            report['conflict_with_opencs2_dropped'] += 1
            conflicts.append(dict(key=list(k), hltv=h['metrics']['headshot_kills'], opencs2=demo[k]))
    return keep, dict(report, opencs2_rows=len(demo)), conflicts
def link_all(pages, matches, maps, excluded, imported):
    captures, rejected, linked = [], [], []
    for kind, ident in sorted(pages):
        if kind != 'match':
            continue
        try:
            series = om.capture_series(ident, pages)
        except (Blocked, InvalidPage, om.Rejected) as exc:
            rejected.append(dict(hltv_match_id=ident, stage='parse', reason=f'{type(exc).__name__}: {exc}'))
            continue
        series['imported_utc'] = imported
        captures.append(series)
        try:
            mid, flip = om.link_local(series, matches, maps, excluded)
        except om.Rejected as exc:
            rejected.append(dict(hltv_match_id=ident, stage='local_series_link', reason=str(exc)))
            continue
        series['local_match_id'], series['local_team_order_flipped'] = mid, flip
        linked.append((series, mid, flip))
    return captures, rejected, linked


def run(out=OUT, base_history=BASE, history_patch=PATCH, manifest_folders=(VERIFIED,), derive_folders=(BULK,),
        opencs2=OPENCS2, report_dir=REPORTS):
    from known_map_recent.features import excluded_map_keys, load_source
    from known_map_recent.hltv_backfill.history import read_patch

    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refuse to overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    imported = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    matches, maps, players, hashes, _ = load_source()
    excluded = excluded_map_keys()
    extra = None if history_patch is None else read_patch(history_patch, matches, maps, players, hashes, excluded)
    base = json.loads(Path(base_history).read_text(encoding='utf-8-sig'))
    base_counts = read_metrics(base_history, players, hashes, extra)[1]
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        stage.mkdir()
        pages, sources, orphans = collect_pages(stage, manifest_folders, derive_folders)
        captures, rejected, linked = link_all(pages, matches, maps, excluded, imported)
        prior = {(r['hltv_player_id'], r['player_id']) for r in base['rows']}
        accepted, skipped = om.link_players(linked, players, prior)
        rows = [headshot_row(*a, imported, pages) for a in accepted]
        rows, opencs2_report, opencs2_conflicts = opencs2_check(rows, opencs2, players, hashes)
        new_captures = []
        merged_rows, merge_counts, not_merged = merge_headshots(base['rows'], rows, new_captures)
        patch = dict(version=VERSION, source_sha256=hashes, metric_source=HS_SOURCE, metric_schema=HS_SCHEMA,
                     imported_utc=imported, rows=rows)
        merged = dict(base, rows=merged_rows, report=dict(base.get('report', {}), browser_headshots_v1=dict(
            base_file=rel(base_history), base_sha256=om.sha256(base_history), headshot_rows=len(rows), **merge_counts)))
        om.write_json(stage / 'capture.json', dict(version=om.CAPTURE_VERSION, imported_utc=imported,
                                                   capture=sources, series=captures, rejected=rejected,
                                                   duplicate_content_new_captures=new_captures))
        om.write_json(stage / 'headshot_patch.json', patch)
        om.write_json(stage / 'metric_history.json', merged)
        report = dict(
            imported_utc=imported, capture_sources=sources, mapstats_pages_without_saved_match_link=orphans,
            match_page_khs_scan=scan_match_khs([*manifest_folders, *derive_folders]),
            series_with_mapstats=len(captures), series_linked=len(linked),
            maps_captured=sum(len(s['maps']) for s in captures), rejected_series=rejected,
            accepted_headshot_player_maps=len(rows), opencs2=opencs2_report, opencs2_conflicts=opencs2_conflicts,
            merge=merge_counts, not_merged=not_merged, skipped_player_maps=skipped,
            skipped_by_reason=dict(Counter(s['reason'] for s in skipped)),
            per_series={s['match_id']: dict(local_match_id=s.get('local_match_id'), maps=len(s['maps']),
                                            headshot_rows=sum(r['hltv_id'] == s['match_id'] for r in rows))
                        for s in captures},
            base_counts=base_counts,
            headshot_patch_counts=read_metrics(stage / 'headshot_patch.json', players, hashes, extra)[1],
            merged_history_counts=read_metrics(stage / 'metric_history.json', players, hashes, extra)[1],
            merged_history_sources=dict(Counter(r['metric_source'] for r in merged_rows)),
            scope='K(hs) only from saved mapstats pages; saved match pages carry no K(hs) and add no headshots')
        report['files_sha256'] = {f.name: om.sha256(f) for f in sorted(stage.iterdir())}
        om.write_json(stage / 'import_report.json', report)
        stage.rename(out)
    if report_dir is not None:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        om.write_json(Path(report_dir) / f'{out.name}_import_report.json', dict(run_dir=rel(out), **report))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=OUT, help='new run directory (must not exist)')
    p.add_argument('--base-history', type=Path, default=BASE)
    p.add_argument('--history-patch', type=Path, default=PATCH, help="validates the base history's patched rows")
    p.add_argument('--manifest-folder', type=Path, action='append', help='folder with its own capture_manifest.json')
    p.add_argument('--derive-folder', type=Path, action='append', help='folder whose manifest is derived offline')
    p.add_argument('--opencs2-history', type=Path, default=OPENCS2, help="cross-check only; 'none' to skip")
    p.add_argument('--report-dir', type=Path, default=REPORTS)
    a = p.parse_args(argv)
    report = run(a.out, a.base_history, a.history_patch, a.manifest_folder or [VERIFIED], a.derive_folder or [BULK],
                 None if str(a.opencs2_history).lower() == 'none' else a.opencs2_history, a.report_dir)
    print(json.dumps({k: v for k, v in report.items() if k not in ('skipped_player_maps', 'capture_sources')},
                     indent=2, ensure_ascii=False))
    return 0 if report['accepted_headshot_player_maps'] else 2


if __name__ == '__main__':
    sys.exit(main())
