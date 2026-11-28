"""Offline tests for the strict browser-saved HLTV K(hs) import (no network)."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from known_map_recent.hltv_backfill.parse import InvalidPage
from known_map_v2_extended import browser_headshots as bh
from known_map_v2_extended import metric_history as mh

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'temp'
HAVE_SAVED = (bh.VERIFIED / 'capture_manifest.json').is_file() and any(bh.BULK.glob('mapstats_*.html'))


def hs_row(**kw):
    r = dict(match_id=1, map_name='ancient', day='2025-08-12', side=1, player_id=7, kills=20, deaths=15,
             adr=80.0, kast=70.0, scores=[13, 10], rounds=23, hltv_id=99, mapstats_id=5, hltv_player_id=70,
             identity_proven_by=[5, 6], identity_as_of='2025-08-12', metrics={'headshot_kills': 9},
             metric_source=mh.HS_SOURCE, metric_schema=mh.HS_SCHEMA, sha256='a' * 64,
             source='https://www.hltv.org/stats/matches/mapstatsid/5/a-vs-b',
             source_url='https://www.hltv.org/stats/matches/mapstatsid/5/a-vs-b',
             match_source_url='https://www.hltv.org/matches/99/a-vs-b', match_sha256='c' * 64,
             captured_utc='2026-11-28T08:00:00+00:00', imported_utc='2026-11-28T09:00:00+00:00')
    r.update(kw)
    return r


def rating_row(**kw):
    r = {c: hs_row()[c] for c in mh.CORE}
    r.update(scores=[13, 10], hltv_id=99, mapstats_id=5, hltv_player_id=70, identity_proven_by=[5, 6],
             identity_as_of='2025-08-12', sha256='b' * 64, source='archived-hltv-html:99.html',
             metrics={'rating': 1.25}, rating_version='3.0', metric_source=mh.RATING_SOURCE)
    r.update(kw)
    return r


class ValidatorTests(unittest.TestCase):
    def test_valid_headshot_row(self):
        self.assertTrue(mh.headshot_row_ok(hs_row()))

    def test_headshot_row_rejections(self):
        bad = dict(
            same_ids=dict(identity_proven_by=[5, 5]), one_id=dict(identity_proven_by=[5]),
            zero_id=dict(identity_proven_by=[0, 6]), late=dict(identity_as_of='2025-08-13'),
            float_hs=dict(metrics={'headshot_kills': 9.0}), hs_gt_k=dict(metrics={'headshot_kills': 21}),
            mixed=dict(metrics={'headshot_kills': 9, 'rating': 1.0}), rated=dict(rating_version='3.0'),
            schema=dict(metric_schema='other'), page_id=dict(mapstats_id=6),
            match_page_as_source=dict(source='https://www.hltv.org/matches/99/a-vs-b',
                                      source_url='https://www.hltv.org/matches/99/a-vs-b'),
            match_id=dict(hltv_id=98), sha=dict(sha256='x'), same_sha=dict(match_sha256='a' * 64),
            local_time=dict(captured_utc='2026-11-28T08:00:00+08:00'),
            captured_after_import=dict(captured_utc='2026-11-28T10:00:00+00:00'), no_time=dict(captured_utc=None))
        for name, change in bad.items():
            with self.subTest(name):
                self.assertFalse(mh.headshot_row_ok(hs_row(**change)))

    def test_multi_row_requires_both_full_components(self):
        good = bh.multi_row(rating_row(), hs_row())
        self.assertTrue(mh.multi_row_ok(good))
        self.assertEqual(good['metrics'], {'rating': 1.25, 'headshot_kills': 9})
        tampered = copy.deepcopy(good)
        tampered['metrics']['headshot_kills'] = 8
        self.assertFalse(mh.multi_row_ok(tampered))
        tampered = copy.deepcopy(good)
        tampered['metric_provenance']['rating']['identity_as_of'] = '2025-09-01'
        tampered['sha256'] = mh.provenance_digest(tampered['metric_provenance'])
        self.assertFalse(mh.multi_row_ok(tampered))
        tampered = copy.deepcopy(good)
        tampered['metric_provenance']['headshot_kills']['kills'] = 21  # core drift between components
        tampered['sha256'] = mh.provenance_digest(tampered['metric_provenance'])
        self.assertFalse(mh.multi_row_ok(tampered))
        tampered = copy.deepcopy(good)
        tampered['sha256'] = 'd' * 64
        self.assertFalse(mh.multi_row_ok(tampered))


class MergeTests(unittest.TestCase):
    def test_new_combined_duplicate_conflict(self):
        rows, counts, skipped = bh.merge_headshots([], [hs_row()])
        self.assertEqual((len(rows), counts, skipped), (1, {'new_headshot_only_rows': 1}, []))
        rows, counts, _ = bh.merge_headshots([rating_row()], [hs_row()])
        self.assertEqual((rows[0]['metric_source'], counts), (mh.MULTI_SOURCE, {'rating_rows_combined_with_headshots': 1}))
        self.assertEqual(rows[0]['metric_provenance']['rating'], rating_row())  # rating kept unchanged
        again, counts, _ = bh.merge_headshots(rows, [hs_row()])
        self.assertEqual((again, counts), (rows, {'duplicate_existing_headshot_rows': 1}))
        with self.assertRaises(ValueError):
            bh.merge_headshots(rows, [hs_row(metrics={'headshot_kills': 8})])
        with self.assertRaises(ValueError):
            bh.merge_headshots([rating_row(), rating_row()], [])

    def test_same_content_new_page_sha_keeps_old_source(self):
        for base in ([hs_row()], bh.merge_headshots([rating_row()], [hs_row()])[0]):
            with self.subTest(base[0]['metric_source']):
                recapture = hs_row(sha256='e' * 64, captured_utc='2026-11-28T08:30:00+00:00')
                evidence = []
                rows, counts, skipped = bh.merge_headshots(base, [recapture], evidence)
                self.assertEqual((rows, counts, skipped), (base, {'duplicate_content_new_capture': 1}, []))
                self.assertEqual(len(evidence), 1)
                self.assertEqual((evidence[0]['kept_sha256'], evidence[0]['new_capture']), ('a' * 64, recapture))
                self.assertEqual(bh.merge_headshots(base, [recapture])[1], {'duplicate_content_new_capture': 1})

    def test_value_core_or_identity_change_still_rejected(self):
        for change in (dict(metrics={'headshot_kills': 8}), dict(kills=21), dict(adr=80.1), dict(day='2025-08-13'),
                       dict(hltv_player_id=71)):
            for base in ([hs_row()], bh.merge_headshots([rating_row()], [hs_row()])[0]):
                with self.subTest(change=change, source=base[0]['metric_source']):
                    with self.assertRaises(ValueError):
                        bh.merge_headshots(base, [hs_row(sha256='e' * 64, **change)], [])

    def test_core_disagreement_not_combined(self):
        rows, counts, skipped = bh.merge_headshots([rating_row(adr=81.0)], [hs_row()])
        self.assertEqual(rows, [rating_row(adr=81.0)])
        self.assertEqual(skipped[0]['reason'], 'base_core_or_identity_differs')


MATCH = ('<html><head><title>x</title><link href="https://www.hltv.org/matches/{mid}/a-vs-b" rel="canonical">'
         '</head><body><a href="/stats/matches/mapstatsid/5/a-vs-b">m</a>'
         '<a href="/stats/matches/mapstatsid/6/b-vs-a">m</a></body></html>')
MAPSTATS = '<html><body><a href="/stats/matches/mapstatsid/{i}/{slug}">t</a></body></html>'


class DeriveManifestTests(unittest.TestCase):
    def folder(self, tmp, mid=99, slug5='a-vs-b'):
        d = Path(tmp)
        (d / 'match_99.html').write_text(MATCH.format(mid=mid), encoding='utf-8')
        (d / 'match_100.html').write_text('<html>no mapstats link</html>', encoding='utf-8')
        (d / 'mapstats_5.html').write_text(MAPSTATS.format(i=5, slug=slug5), encoding='utf-8')
        (d / 'mapstats_6.html').write_text(MAPSTATS.format(i=6, slug='b-vs-a'), encoding='utf-8')
        (d / 'mapstats_7.html').write_text(MAPSTATS.format(i=7, slug='c'), encoding='utf-8')
        return d

    def test_urls_from_page_links_and_hashes_verify(self):
        from known_map_v2_extended import offline_mapstats as om
        with TemporaryDirectory(dir=TMP) as tmp:
            d = self.folder(tmp)
            man, orphans = bh.derive_manifest(d)
            self.assertEqual(orphans, [7])
            self.assertEqual([f['file'] for f in man['files']], ['match_99.html', 'mapstats_5.html', 'mapstats_6.html'])
            self.assertEqual(man['files'][1]['url'], 'https://www.hltv.org/stats/matches/mapstatsid/5/a-vs-b')
            (d / 'm.json').write_text(json.dumps(man), encoding='utf-8')
            _, pages = om.read_manifest(d, d / 'm.json')
            self.assertEqual(set(pages), {('match', 99), ('mapstats', 5), ('mapstats', 6)})

    def test_wrong_canonical_or_unlinked_mapstats_page_rejected(self):
        with TemporaryDirectory(dir=TMP) as tmp:
            with self.assertRaises(InvalidPage):
                bh.derive_manifest(self.folder(tmp, mid=98))
        with TemporaryDirectory(dir=TMP) as tmp:
            with self.assertRaises(InvalidPage):
                bh.derive_manifest(self.folder(tmp, slug5='other-slug'))


@unittest.skipUnless(HAVE_SAVED, 'saved HLTV pages not present')
class SavedTests(unittest.TestCase):
    def test_saved_match_pages_carry_no_khs(self):
        scan = bh.scan_match_khs([bh.BULK, bh.VERIFIED])
        self.assertEqual(scan['match_pages_with_khs_markup'], 0)

    def test_end_to_end(self):
        from known_map_recent.features import load_source
        with TemporaryDirectory(dir=TMP) as tmp:
            out = Path(tmp) / 'run'
            rep = bh.run(out, report_dir=None)
            self.assertGreater(rep['accepted_headshot_player_maps'], 0)
            self.assertEqual(rep['opencs2'].get('conflict_with_opencs2_dropped', 0), len(rep['opencs2_conflicts']))
            patch = json.loads((out / 'headshot_patch.json').read_text(encoding='utf-8'))
            self.assertTrue(all(r['source'].startswith('https://www.hltv.org/stats/matches/mapstatsid/')
                                and set(r['metrics']) == {'headshot_kills'} for r in patch['rows']))
            hist = json.loads((out / 'metric_history.json').read_text(encoding='utf-8'))
            base = json.loads(bh.BASE.read_text(encoding='utf-8'))
            self.assertEqual(rep['merged_history_counts']['rating'], rep['base_counts']['rating'])
            ratings = {tuple(r[c] for c in mh.KEY): r['metrics']['rating'] for r in base['rows']}
            self.assertTrue(all(ratings[tuple(r[c] for c in mh.KEY)] == r['metrics']['rating']
                                for r in hist['rows'] if 'rating' in r['metrics']))
            with self.assertRaises(FileExistsError):
                bh.run(out, report_dir=None)


if __name__ == '__main__':
    unittest.main()
