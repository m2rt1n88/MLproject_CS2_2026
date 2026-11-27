"""Plan v2 backfill from local indexes and verified histories; makes no site requests."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

from known_map_recent.features import excluded_map_keys, load_source
from known_map_recent.hltv_backfill.history import archive_signatures, job_signature, local_series, read_patch
from known_map_recent.hltv_backfill.parse import key
from known_map_recent.hltv_backfill.store import ARCHIVE, BASE, STORE, page_path
from known_map_v2_extended.build import HOME, ROSTERS, START, STOP, WINDOW, five
from known_map_v2_extended.metric_history import METRICS, read_metrics
from known_map_v2_extended.opencs2_headshots import read_history as read_headshots


def recorded_players(players, extra=None):
    """Match the input eligibility of the v2 reconstruction."""
    cases = defaultdict(list)
    for table in (players, extra):
        if table is None:
            continue
        for p in table.itertuples(index=False):
            if 0 < p.rounds and 0 <= p.kills <= 5*p.rounds and 0 <= p.deaths <= p.rounds:
                day = str(p.date if table is players else p.day)[:10]
                cases[p.player_id, p.map_name].append((day, int(p.match_id), p.kills))
    return {k: sorted(v) for k, v in cases.items()}


def candidate_jobs(db, archive, matches):
    local, archive_counts = local_series(matches), archive_signatures(archive)
    result = {}
    player_maps = defaultdict(dict)
    for row in db.execute('SELECT id,name,mapstats_id FROM accepted_players'):
        player_maps[row['id']][row['name']] = row['mapstats_id']
    for job in db.execute('''SELECT j.*,s.scores AS page_scores FROM jobs j
                             LEFT JOIN accepted_scores s ON s.id=j.id ORDER BY j.id'''):
        sk = job_signature(job)
        found = local.get(sk, ())
        if len(found) != 1 or archive_counts[sk] != 1:
            continue
        m, = found
        source = job['page_scores'] or job['archive_scores']
        evidence = 'checked_page' if job['page_scores'] else ('archive_scores' if source else 'series_only')
        scores = json.loads(source) if source else {}
        if key(m.team1_key) != key(job['team_a']):
            scores = {mp: totals[::-1] for mp, totals in scores.items()}
        result[int(m.match_id)] = (dict(hltv_id=job['id'], match_url=BASE + page_path(job), evidence=evidence,
                                         scoreboard_map_ids=player_maps[job['id']]), scores)
    return result


def build_plan(matches, maps, players, *, extra=None, headshots=None, history=None, jobs=None):
    headshots = headshots or {}
    history = history or {}
    jobs = jobs or {}
    recorded = recorded_players(players, extra)
    target_maps = (maps[maps.date.ge(START) & maps.date.lt(STOP)]
                   .groupby('match_id').map_name.agg(sorted).to_dict())
    local_scores = {(r.match_id, r.map_name): [r.score1, r.score2] for r in maps.itertuples(index=False)}
    job_conflicts = {m for (m, mp), score in local_scores.items() if m in jobs and jobs[m][1] and jobs[m][1].get(mp) != score}
    used_player_maps = set()
    impact, single, all_target = defaultdict(Counter), defaultdict(Counter), Counter()
    complete, eligible = Counter(), Counter()
    rosters, hist_map_dates = {}, {}
    ordered = matches[matches.date.lt(STOP)].sort_values(['date', 'match_id'])
    for stamp, group in ordered.groupby(ordered.date.dt.normalize()):
        day, start = stamp.date().isoformat(), (stamp.date() - timedelta(days=WINDOW)).isoformat()
        today = list(group.itertuples(index=False))
        for m in today:
            ts = m.team1_key, m.team2_key
            sides = [rosters.get(team, ()) for team in ts]
            if set(sides[0]) & set(sides[1]):
                continue
            for mp in target_maps.get(m.match_id, ()):
                for ids in sides:
                    if not ids:
                        continue
                    boxes, missing = set(), defaultdict(set)
                    complete_players, with_kills = 0, 0
                    for pid in ids:
                        events = recorded.get((pid, mp), ())
                        earlier = events[bisect_left(events, (start,)):bisect_left(events, (day,))]
                        complete_players += bool(earlier)
                        with_kills += any(kills > 0 for _, _, kills in earlier)
                        for event_day, mid, kills in earlier:
                            pk, mk = (mid, mp, pid), (mid, mp)
                            hist_map_dates[mk] = event_day
                            boxes.add(mk)
                            used_player_maps.add(pk)
                            values = history.get(pk, {})
                            for metric in METRICS:
                                if metric not in values and not (metric == 'headshot_kills' and (kills == 0 or pk in headshots)):
                                    missing[metric].add(mk)
                    all_target.update(boxes)
                    if complete_players != 5:
                        continue
                    for metric in METRICS:
                        if metric == 'headshot_kills' and with_kills != 5:
                            continue
                        eligible[metric] += 1
                        remaining = missing[metric]
                        if not remaining:
                            complete[metric] += 1
                        if len(remaining) == 1:
                            single[next(iter(remaining))][metric] += 1
                        for mk in remaining:
                            impact[mk][metric] += 1
        for r in today:
            for team, cols in zip((r.team1_key, r.team2_key), ROSTERS):
                if ids := five(r, cols):
                    rosters[team] = ids
    rows = []
    for mk in sorted(hist_map_dates):
        mid, mp = mk
        local_score = local_scores.get(mk)
        job, scores = jobs.get(mid, (None, {}))
        if mid in job_conflicts or (scores and mp not in scores):
            job = None
        if job:
            job = {k: v for k, v in job.items() if k != 'scoreboard_map_ids'} | {
                'cached_mapstats_id': job.get('scoreboard_map_ids', {}).get(mp)}
        rows.append(dict(local_match_id=mid, map=mp, day=hist_map_dates[mk], score=local_score, candidate=job,
                         target_team_uses=all_target[mk], possible_affected_team_windows={k: impact[mk][k] for k in METRICS},
                         possible_completions_if_only_this_map_added={k: single[mk][k] for k in METRICS}))
    def order(r):
        # Prefer immediate Rating 3.0 gains; it is the only verified current scoreboard source.
        immediate, overall = r['possible_completions_if_only_this_map_added'], r['possible_affected_team_windows']
        evidence = (r['candidate'] or {}).get('evidence')
        return (r['candidate'] is None, -immediate['rating'], -sum(immediate.values()),
                evidence == 'series_only', -overall['rating'], -sum(overall.values()), r['day'], r['local_match_id'], r['map'])
    rows.sort(key=order)
    rating_batch = {}
    for r in rows:
        c = r['candidate']
        advances = r['possible_affected_team_windows']['rating']
        if c and c['cached_mapstats_id'] is None and advances:
            rating_batch.setdefault(c['hltv_id'], []).append(r)
    shortlist = []
    for records in list(rating_batch.values())[:12]:
        c = records[0]['candidate']
        shortlist.append(dict(hltv_id=c['hltv_id'], match_url=c['match_url'], day=records[0]['day'],
                              priority_maps=[r['map'] for r in records], evidence=c['evidence']))
    return dict(target_maps=sum(map(len, target_maps.values())),
                used_history_player_maps=len(used_player_maps), used_history_maps=len(rows),
                used_history_series=len({r['local_match_id'] for r in rows}),
                linked_candidate_maps=sum(r['candidate'] is not None for r in rows),
                linked_candidate_series=len({r['candidate']['hltv_id'] for r in rows if r['candidate']}),
                score_conflict_series=len(job_conflicts),
                current_complete_team_windows={k: complete[k] for k in METRICS},
                eligible_team_windows={k: eligible[k] for k in METRICS},
                suggested_uncached_rating_pilot=shortlist, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--db', type=Path, default=STORE)
    p.add_argument('--snapshot', type=Path, default=HOME / 'data/five_metrics_v1')
    p.add_argument('--out', type=Path, required=True)
    options = p.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    matches, maps, players, hashes, _ = load_source()
    extra = read_patch(options.snapshot/'hltv_history_patch.json', matches, maps, players, hashes, excluded_map_keys())
    shots = read_headshots(options.snapshot/'headshot_history.json', players, hashes)
    metrics, _ = read_metrics(options.snapshot/'metric_history.json', players, hashes, extra)
    with closing(sqlite3.connect(options.db.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        jobs = candidate_jobs(db, options.archive, matches)
    plan = build_plan(matches, maps, players, extra=extra, headshots=shots, history=metrics, jobs=jobs)
    plan['source_sha256'] = hashes
    plan['note'] = ('Candidates are not verified pages. Possible gains assume all missing player values on the named '
                    'map can be verified, consider recorded local history only, and count team-map examples (not matches). '
                    'Do not use these counts as source coverage or add any unlabeled targets.')
    with options.out.open('x', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in plan.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
