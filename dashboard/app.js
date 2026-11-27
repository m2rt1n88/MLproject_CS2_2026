'use strict';

const byId = id => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function probability(predictions, a, b) {
  const forward = predictions[`${a}|${b}`];
  if (forward !== undefined) return forward;
  const reverse = predictions[`${b}|${a}`];
  return typeof reverse === 'number' ? 1 - reverse : null;
}

function teamLogo(team) {
  const tile = element('div', 'team-logo');
  tile.setAttribute('aria-hidden', 'true'); // The full team name is displayed next to the logo.
  const initials = team.name.split(/\s+/).map(word => word[0]).join('');
  const fallback = element('span', 'logo-fallback', (initials.length > 1 ? initials : team.name).slice(0, 3).toUpperCase());
  fallback.hidden = true;
  const img = element('img');
  img.alt = '';
  img.width = img.height = 56;
  img.decoding = 'async';
  img.addEventListener('error', () => {
    img.hidden = true;
    fallback.hidden = false;
  }, {once: true});
  img.src = 'logos/' + encodeURIComponent(team.id) + '.png';
  tile.append(img, fallback);
  return tile;
}

function teamSide(team, tenths, favorite) {
  const side = element('div', favorite ? 'side favorite' : 'side');
  const label = element('div', 'team-label');
  const percent = tenths === null ? '—' : (tenths / 10).toFixed(1) + '%';
  label.append(element('div', 'name', team.name), element('div', 'percent', percent));
  side.append(teamLogo(team), label);
  return side;
}

function matchup(a, b, p) {
  // Round one side and use its complement so the labels always add up to 100%.
  const left = p === null ? null : Math.round(p * 1000);
  const right = left === null ? null : 1000 - left;
  const vs = element('span', 'match-vs', 'VS');
  vs.setAttribute('aria-hidden', 'true');
  const pair = element('div', 'match-pair');
  pair.append(teamSide(a, left, p !== null && p > .5), vs,
    teamSide(b, right, p !== null && p < .5));
  if (p === null) return [pair, element('p', 'empty', 'Not enough match history to make a prediction.')];
  const rail = element('div', 'rail');
  rail.setAttribute('aria-hidden', 'true');
  [p, 1 - p].forEach(value => {
    const segment = element('span', value > .5 ? 'favorite' : '');
    segment.style.flexGrow = String(value);
    rail.append(segment);
  });
  return [pair, rail];
}

function option(value, label) {
  const node = element('option', '', label);
  node.value = value;
  return node;
}

function fixtureCard(fixture, a, b, p) {
  const {id, day, time, kickoff, stage} = fixture;
  const when = element('time', '', day + ' · ' + kickoff + ' SGT');
  when.dateTime = time;
  const link = element('a', '', 'Stage ' + stage + ' ↗');
  link.href = 'https://www.hltv.org/matches/' + id + '/match';
  link.setAttribute('aria-label', 'Match details: ' + a.name + ' vs ' + b.name);

  const meta = element('div', 'fixture-meta');
  meta.append(when, link);
  const card = element('article', 'fixture');
  card.append(meta, ...matchup(a, b, p));
  return card;
}

function setup(data) {
  const {roster, predictions, source, scheduleSnapshot} = data;
  const teams = new Map(roster.map(team => [team.id, team]));
  const teamPair = (a, b) => [teams.get(a), teams.get(b), probability(predictions, a, b)];
  byId('team-count').textContent = roster.length;
  byId('fixture-count').textContent = data.fixtures.length;
  byId('event-source').href = source;

  const timeZone = 'Asia/Singapore';
  const dayFormat = new Intl.DateTimeFormat('en-CA', {timeZone, year: 'numeric', month: '2-digit', day: '2-digit'});
  const timeFormat = new Intl.DateTimeFormat('en-GB', {timeZone, hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
  const fixtures = data.fixtures.map(fixture => {
    const date = new Date(fixture.time);
    return {...fixture, day: dayFormat.format(date), kickoff: timeFormat.format(date)};
  });
  const days = [...new Set(fixtures.map(({day}) => day))];
  const daySelect = byId('match-day');
  const fixtureList = byId('fixtures');
  daySelect.append(option('all', 'All listed matches'), ...days.map(day => option(day, day)));
  daySelect.value = days.includes(scheduleSnapshot) ? scheduleSnapshot : days.at(-1) || 'all';

  function renderFixtures() {
    const day = daySelect.value;
    const visible = day === 'all' ? fixtures : fixtures.filter(fixture => fixture.day === day);
    const cards = visible.map(f => fixtureCard(f, ...teamPair(f.a, f.b)));
    fixtureList.replaceChildren(...(cards.length ? cards : [element('p', 'empty', 'No matches listed.')]));
  }

  const selectA = byId('team-a');
  const selectB = byId('team-b');
  for (const stage of [1, 2, 3]) {
    const group = element('optgroup');
    group.label = 'Starting in Stage ' + stage;
    group.append(...roster.filter(t => t.stage === stage).map(t => option(t.id, t.name)));
    selectA.append(group);
    selectB.append(group.cloneNode(true));
  }
  const featured = fixtures.find(f => f.day === daySelect.value) || fixtures[0];
  selectA.value = featured?.a || roster[0].id;
  selectB.value = featured?.b || roster[1].id;

  const comparison = byId('comparison');
  comparison.className = 'compare-result';
  function renderComparison() {
    const {value: a} = selectA;
    const {value: b} = selectB;
    const result = a === b ? [element('p', 'empty', 'Select two different teams.')] : matchup(...teamPair(a, b));
    comparison.replaceChildren(...result);
  }

  daySelect.addEventListener('change', renderFixtures);
  for (const select of [selectA, selectB]) select.addEventListener('change', renderComparison);
  renderFixtures();
  renderComparison();
}

if (window.SINGAPORE_DATA) setup(window.SINGAPORE_DATA);
else byId('load-error').hidden = false;
