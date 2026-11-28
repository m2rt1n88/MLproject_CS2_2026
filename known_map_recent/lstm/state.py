"""Chronological team/map/player history. Query a whole day first, then ingest that day."""
from collections import defaultdict, deque
from datetime import date as Date
import math

import numpy as np
import pandas as pd

ROSTERS = [[f'team{s}_player{i}_id' for i in range(1, 6)] for s in (1, 2)]
SEQ_LEN = 20
ELO_K = 32.0
PRIOR_ADR, PRIOR_KAST, PRIOR_KD = 75.0, 70.0, 0.0
PLAYER_SHRINK_ROUNDS = 80.0
TEAM_SHRINK, MAP_SHRINK = 10.0, 5.0
STATIC_NAMES = (
    'map_winrate_shrunk', 'map_minus_overall_winrate', 'map_margin_shrunk', 'log_map_games', 'log_overall_games',
    'map_cold_start', 'overall_cold_start', 'log_days_since_map', 'log_days_since_any', 'overall_elo', 'map_elo',
    'map_elo_minus_overall', 'player_map_adr', 'player_map_kast', 'player_map_kd', 'player_map_adr_specialty',
    'player_map_kast_specialty', 'player_map_kd_specialty', 'player_coverage', 'player_map_coverage',
    'log_player_map_rounds', 'log_roster_age_days', 'roster_missing',
)
SEQ_NAMES = ('won_pm1', 'round_margin', 'opponent_elo', 'log_days_gap', 'same_target_map', 'team_adr', 'team_kast',
             'player_stats_missing')
NO_GAP_DAYS = 365.0
# Optional per-player metric features. Career-long (not a 90-day window), partial coverage is explicit.
# Rate is the career-wide mean: sum(hs)/sum(kills) for headshots, sum(rating*rounds)/sum(rounds) for rating3.
# Specialty is the per-map rate shrunk to the career mean (map deviation).
HS_PRIOR, HS_SHRINK_KILLS, HS_SCALE = 0.45, 40.0, 0.1
RATING_PRIOR, RATING_SHRINK_ROUNDS, RATING_SCALE = 1.0, 80.0, 0.2
HS_NAMES = ('hs_overall_rate', 'hs_map_specialty', 'hs_coverage', 'hs_map_coverage', 'log_hs_kills', 'hs_missing')
RATING_NAMES = ('rating3_overall_mean', 'rating3_map_specialty', 'rating3_coverage', 'rating3_map_coverage',
                'log_rating3_rounds', 'rating3_missing')
FEATURE_SETS = {
    'base': dict(headshots=False, rating3=False),
    'headshots_only': dict(headshots=True, rating3=False),
    'rating3_headshots': dict(headshots=True, rating3=True),
}


def static_names(feature_set='base'):
    spec = FEATURE_SETS[feature_set]
    return STATIC_NAMES + (HS_NAMES if spec['headshots'] else ()) + (RATING_NAMES if spec['rating3'] else ())


def as_day(value):
    if isinstance(value, Date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def norm_team(name):
    return str(name).strip().lower()


def norm_map(name):
    return str(name).strip().lower()


def validate_lineup(lineup):
    """A provided lineup must be exactly 5 unique positive integer IDs (unknown IDs are allowed)."""
    if isinstance(lineup, (str, bytes)) or not hasattr(lineup, '__len__') or len(lineup) != 5:
        raise ValueError(f'lineup must contain exactly 5 player IDs, got {lineup!r}')
    ids = []
    for i in lineup:
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or int(i) <= 0:
            raise ValueError(f'lineup player IDs must be positive integers, got {i!r}')
        ids.append(int(i))
    if len(set(ids)) != 5:
        raise ValueError(f'lineup player IDs must be unique, got {ids}')
    return tuple(ids)


class HistoryState:
    """All values returned by query() depend only on days already passed to ingest_day()."""

    def __init__(self, feature_set='base'):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f'unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}')
        self.feature_set, self.spec = feature_set, FEATURE_SETS[feature_set]
        self.static_names = static_names(feature_set)
        self.rating = defaultdict(lambda: np.zeros(3))   # (pid,map|None) -> sum rating*rounds, rounds, maps
        self.hs = defaultdict(lambda: np.zeros(3))       # (pid,map|None) -> sum headshot kills, sum kills, maps
        self.last_ingested = None
        self.team_games = defaultdict(lambda: np.zeros(2))          # team -> wins, games
        self.map_games = defaultdict(lambda: np.zeros(3))           # (team,map) -> wins, games, margin sum
        self.last_map_day, self.last_any_day = {}, {}
        self.elo, self.map_elo = defaultdict(lambda: 1500.0), {}
        self.seq = defaultdict(lambda: deque(maxlen=SEQ_LEN))
        self.player = defaultdict(lambda: np.zeros(5))              # (pid,map|None) -> dmg,rounds,kast*r,kills,deaths
        self.roster = {}                                            # team -> (ids, day)

    # ----- queries -----
    def _player_form(self, pid, map_name):
        dmg, rounds, kast, kills, deaths = self.player[pid, None]
        r0 = PLAYER_SHRINK_ROUNDS
        o_adr = (dmg + PRIOR_ADR * r0) / (rounds + r0)
        o_kast = (kast + PRIOR_KAST * r0) / (rounds + r0)
        o_kd = (kills - deaths + PRIOR_KD * r0) / (rounds + r0)
        m_dmg, m_rounds, m_kast, m_kills, m_deaths = self.player[pid, map_name]
        m_adr = (m_dmg + o_adr * r0) / (m_rounds + r0)
        m_kast2 = (m_kast + o_kast * r0) / (m_rounds + r0)
        m_kd = (m_kills - m_deaths + o_kd * r0) / (m_rounds + r0)
        return np.array([m_adr, m_kast2, m_kd, m_adr - o_adr, m_kast2 - o_kast, m_kd - o_kd]), rounds, m_rounds

    def _metric_side(self, table, ids, map_name, prior, shrink, scale):
        """Mean career rate over players with an observed, defined history; map specialty is shrunk to it.

    Players without a defined rate are left out (coverage drops); if nobody is observed the rate is the
    prior (centred value 0) and the `*_missing` mask is 1, which differs from an observed rate of 0.
    """
        rates, specs, weight, seen, seen_map = [], [], 0.0, 0, 0
        for pid in ids:
            num, den, _ = table[pid, None]
            if den <= 0:          # no history, or only events with an undefined rate (e.g. 0 kills)
                continue
            seen += 1
            weight += den
            overall = num / den     # rating: sum(r*rounds)/sum(rounds); headshots: sum(hs)/sum(kills)
            rates.append(overall)
            m_num, m_den, _ = table[pid, map_name]
            if m_den > 0:
                seen_map += 1
                specs.append((m_num + overall * shrink) / (m_den + shrink) - overall)
        rate = (np.mean(rates) - prior) / scale if rates else 0.0
        spec = np.mean(specs) / scale if specs else 0.0
        values = [rate, spec, seen / 5, seen_map / 5, math.log1p(weight / 5), float(not rates)]
        return values, dict(players=seen, map_players=seen_map, coverage=seen / 5, map_coverage=seen_map / 5,
                            raw_mean=float(np.mean(rates)) if rates else None)

    def player_metric(self, kind, pid, map_name=None):
        """Raw (numerator, denominator, maps) for tests and audits; kind is 'headshots' or 'rating3'."""
        table = self.hs if kind == 'headshots' else self.rating
        return tuple(float(x) for x in table[int(pid), None if map_name is None else norm_map(map_name)])

    def _lineup(self, team, lineup, day):
        if lineup is not None:
            return validate_lineup(lineup), 'provided', None
        if team in self.roster:
            ids, since = self.roster[team]
            return ids, 'proxy', since
        return (), 'none', None

    def query(self, team, map_name, day, lineup=None):
        """Return (static vector, sequence numeric array [T,F], sequence map names, metadata)."""
        team, map_name, day = norm_team(team), norm_map(map_name), as_day(day)
        if self.last_ingested is not None and day <= self.last_ingested:
            raise ValueError(f'query day {day} is not after ingested history {self.last_ingested}')
        wins, games = self.team_games[team]
        overall_wr = (wins + 0.5 * TEAM_SHRINK) / (games + TEAM_SHRINK)
        m_wins, m_games, m_margin = self.map_games[team, map_name]
        map_wr = (m_wins + overall_wr * MAP_SHRINK) / (m_games + MAP_SHRINK)
        margin = m_margin / (m_games + MAP_SHRINK)
        gap = lambda d: math.log1p((day - d).days if d is not None else NO_GAP_DAYS)
        elo = self.elo[team]
        map_elo = self.map_elo.get((team, map_name), elo)
        ids, source, since = self._lineup(team, lineup, day)
        forms, known, known_map, map_rounds = [], 0, 0, 0.0
        for pid in ids[:5]:
            form, rounds, m_rounds = self._player_form(pid, map_name)
            if rounds > 0:
                known += 1
                forms.append(form)
            if m_rounds > 0:
                known_map += 1
                map_rounds += m_rounds
        # Missing or unknown slots fall back to the global prior.
        prior = np.array([PRIOR_ADR, PRIOR_KAST, PRIOR_KD, 0.0, 0.0, 0.0])
        forms += [prior] * (5 - len(forms))
        player = np.mean(forms, axis=0)
        roster_age = math.log1p((day - since).days) if since is not None else 0.0
        static = np.array([
            map_wr, map_wr - overall_wr, margin, math.log1p(m_games), math.log1p(games), float(m_games == 0),
            float(games == 0), gap(self.last_map_day.get((team, map_name))), gap(self.last_any_day.get(team)),
            (elo - 1500) / 400, (map_elo - 1500) / 400, (map_elo - elo) / 400, (player[0] - PRIOR_ADR) / 15,
            (player[1] - PRIOR_KAST) / 10, player[2], player[3] / 15, player[4] / 10, player[5], known / 5,
            known_map / 5, math.log1p(map_rounds / 5), roster_age, float(not ids)], dtype=np.float32)
        metric_meta = {}
        extra = []
        if self.spec['headshots']:
            values, metric_meta['headshots'] = self._metric_side(self.hs, ids[:5], map_name, HS_PRIOR,
                                                                 HS_SHRINK_KILLS, HS_SCALE)
            extra += values
        if self.spec['rating3']:
            values, metric_meta['rating3'] = self._metric_side(self.rating, ids[:5], map_name, RATING_PRIOR,
                                                               RATING_SHRINK_ROUNDS, RATING_SCALE)
            extra += values
        if extra:
            static = np.concatenate([static, np.asarray(extra, dtype=np.float32)])
        seq = list(self.seq[team])
        numeric = np.zeros((len(seq), len(SEQ_NAMES)), dtype=np.float32)
        names = []
        for i, (d, won, mg, opp, name, adr, kast, missing) in enumerate(seq):
            numeric[i] = (won, mg, opp, math.log1p((day - d).days), float(name == map_name), adr, kast, missing)
            names.append(name)
        meta = dict(team=team, team_known=bool(games > 0), map_name=map_name, lineup_source=source,
                    lineup_ids=list(ids),
                    player_coverage=known / 5, player_map_coverage=known_map / 5,
                    team_map_games=int(m_games), team_games=int(games), sequence_length=len(seq),
                    feature_set=self.feature_set, metric_coverage=metric_meta)
        return static, numeric, names, meta

    # ----- updates -----
    def ingest_day(self, day, matches, maps, players, metrics=None):
        """Add one day's results, player lines, lineups and (optional) metric events, all from `day`.

        `metrics` columns: date, map_name, player_id, rounds, kills, rating, headshot_kills; NaN = not observed.
        Each field is ingested independently; fields not enabled by the feature set are ignored.
        """
        day = as_day(day)
        if self.last_ingested is not None and day <= self.last_ingested:
            raise ValueError('days must be ingested in increasing order')
        frames = (matches, maps, players) if metrics is None else (matches, maps, players, metrics)
        for frame in frames:
            if len(frame) and not frame.date.dt.date.eq(day).all():
                raise ValueError('ingest_day received rows from another day')
        if metrics is not None and len(metrics):
            self._ingest_metrics(metrics)
        lines = defaultdict(list)
        for r in players.itertuples(index=False):
            lines[r.match_id, norm_map(r.map_name), int(r.side)].append(r)
        keys = matches.set_index('match_id')[['team1_key', 'team2_key']]
        for r in maps.sort_values('match_id', kind='stable').itertuples(index=False):
            a, b = (norm_team(t) for t in keys.loc[r.match_id])
            name = norm_map(r.map_name)
            total = r.score1 + r.score2
            mg = (r.score1 - r.score2) / total if total > 0 else 0.0
            won = int(r.team1_won_map)
            ea, eb = self.elo[a], self.elo[b]
            ma, mb = self.map_elo.get((a, name), ea), self.map_elo.get((b, name), eb)
            for team, opp, w, m, side in ((a, eb, won, mg, 1), (b, ea, 1 - won, -mg, 2)):
                rows = lines[r.match_id, name, side]
                adr = (np.mean([x.adr for x in rows]) - PRIOR_ADR) / 15 if rows else 0.0
                kast = (np.mean([x.kast for x in rows]) - PRIOR_KAST) / 10 if rows else 0.0
                self.seq[team].append((day, 2.0 * w - 1, m, (opp - 1500) / 400, name, adr, kast, float(not rows)))
                self.team_games[team] += (w, 1)
                self.map_games[team, name] += (w, 1, m)
                self.last_map_day[team, name] = day
                self.last_any_day[team] = day
            exp = 1 / (1 + 10 ** ((eb - ea) / 400))
            self.elo[a], self.elo[b] = ea + ELO_K * (won - exp), eb - ELO_K * (won - exp)
            exp_m = 1 / (1 + 10 ** ((mb - ma) / 400))
            self.map_elo[a, name], self.map_elo[b, name] = ma + ELO_K * (won - exp_m), mb - ELO_K * (won - exp_m)
        for r in players.itertuples(index=False):
            vals = [r.adr, r.kast, r.kills, r.deaths, r.rounds]
            if not np.isfinite(vals).all() or r.rounds <= 0:
                continue
            data = np.array([r.adr * r.rounds, r.rounds, r.kast * r.rounds, r.kills, r.deaths])
            pid = int(r.player_id)
            self.player[pid, norm_map(r.map_name)] += data
            self.player[pid, None] += data
        for r in matches.itertuples(index=False):
            for team, cols in zip((r.team1_key, r.team2_key), ROSTERS):
                ids = [getattr(r, c) for c in cols]
                if all(pd.notna(i) for i in ids) and len(set(ids)) == 5:
                    ids = tuple(int(i) for i in ids)
                    team = norm_team(team)
                    if self.roster.get(team, (None,))[0] != ids:
                        self.roster[team] = (ids, day)
        self.last_ingested = day

    def _ingest_metrics(self, metrics):
        for r in metrics.itertuples(index=False):
            pid, name = int(r.player_id), norm_map(r.map_name)
            if self.spec['headshots'] and pd.notna(r.headshot_kills):
                hs, kills = float(r.headshot_kills), float(r.kills)
                if not (math.isfinite(kills) and 0 <= hs <= kills):
                    raise ValueError(f'invalid headshot event for player {pid}')
                # 0 kills: the event is counted as a map, but contributes nothing to the rate denominator.
                for key in ((pid, name), (pid, None)):
                    self.hs[key] += (hs, kills, 1)
            if self.spec['rating3'] and pd.notna(r.rating):
                rating, rounds = float(r.rating), float(r.rounds)
                if not (math.isfinite(rating) and rating >= 0 and rounds > 0):
                    raise ValueError(f'invalid rating event for player {pid}')
                for key in ((pid, name), (pid, None)):
                    self.rating[key] += (rating * rounds, rounds, 1)
