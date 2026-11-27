import hashlib
from itertools import combinations
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / 'dashboard'


class SingaporeDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = (DASH / 'predictions.js').read_text(encoding='utf-8').strip()
        assert script.startswith('window.SINGAPORE_DATA = ') and script.endswith(';')
        cls.data = json.loads(script.removeprefix('window.SINGAPORE_DATA = ').removesuffix(';'))

    def test_event_matches_source_snapshot(self):
        event = json.loads((DASH / 'event.json').read_text(encoding='utf-8'))
        self.assertEqual({key: self.data[key] for key in event}, event)
        ids = {t['id'] for t in event['roster']}
        self.assertEqual(len(ids), 32)
        for match in event['fixtures']:
            self.assertNotEqual(match['a'], match['b'])
            self.assertTrue({match['a'], match['b']} <= ids)
            self.assertEqual(set(match), {'id', 'a', 'b', 'stage', 'time'})

    def test_all_teams_have_local_png_crests(self):
        for team in self.data['roster']:
            with self.subTest(team=team['name']):
                image = (DASH / 'logos' / (team['id'] + '.png')).read_bytes()
                self.assertEqual(image[:16], b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR')
                self.assertGreater(int.from_bytes(image[16:20], 'big'), 0)
                self.assertGreater(int.from_bytes(image[20:24], 'big'), 0)

    def test_every_pair_has_exactly_one_prediction(self):
        predictions = self.data['predictions']
        ids = [t['id'] for t in self.data['roster']]
        self.assertEqual(set(predictions), {'|'.join(pair) for pair in combinations(ids, 2)})
        for key, probability in predictions.items():
            has_history = all(self.data['teamHistory'][i]['matches'] > 0 for i in key.split('|'))
            if has_history:
                self.assertIsInstance(probability, float)
                self.assertTrue(0 <= probability <= 1)
            else:
                self.assertIsNone(probability)

    def test_frozen_model_and_history_provenance(self):
        weight = ROOT / 'models/m2/m2_clean_retuned.joblib'
        self.assertEqual(self.data['model'], 'M2')
        self.assertEqual(self.data['modelSha256'], hashlib.sha256(weight.read_bytes()).hexdigest())
        self.assertEqual(self.data['predictionDay'], self.data['start'])
        self.assertLess(self.data['historyThrough'], self.data['predictionDay'])
        for record in self.data['teamHistory'].values():
            if record['lastMatch']:
                self.assertLessEqual(record['lastMatch'], self.data['historyThrough'])
