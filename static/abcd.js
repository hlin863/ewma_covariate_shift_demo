'use strict';
// Only handles the existing Eq. (1)/(5)/(8) display; no numerical recomputation.
const data = JSON.parse(document.getElementById('abcd-data').textContent);
const ns = 'http://www.w3.org/2000/svg';
const svgLoss = document.getElementById('loss-chart');
const svgBound = document.getElementById('bound-chart');
const chooser = document.getElementById('inspect-k');

function shape(root, tag, attrs, value) {
  const element = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([key, item]) => element.setAttribute(key, String(item)));
  if (value !== undefined) element.textContent = String(value);
  root.appendChild(element);
  return element;
}
function frame(root, count, min, max, trueChange, selected) {
  root.replaceChildren();
  const px = i => 54 + (i / count) * 856;
  const py = v => 207 - (175 * (v - min) / Math.max(1e-15, max - min));
  for (let i = 0; i <= 4; i += 1) {
    const value = min + (max - min) * (i / 4), y = py(value);
    shape(root, 'line', {x1: 54, x2: 910, y1: y, y2: y, class: 'chart-grid'});
    shape(root, 'text', {x: 46, y: y + 4, 'text-anchor': 'end', class: 'chart-axis-label'}, value.toPrecision(2));
  }
  shape(root, 'line', {x1: px(trueChange), x2: px(trueChange), y1: 20, y2: 207, class: 'chart-true-line'});
  shape(root, 'line', {x1: px(selected), x2: px(selected), y1: 20, y2: 207, class: 'chart-selected-line'});
  shape(root, 'text', {x: 54, y: 235, class: 'chart-axis-label'}, '0');
  shape(root, 'text', {x: 910, y: 235, 'text-anchor': 'end', class: 'chart-axis-label'}, String(count));
  return {px, py};
}
function render() {
  const k = Number(chooser.value);
  const row = data.splits.find(item => item.k === k);
  if (!row) return;
  const losses = data.observations.map(item => item.loss);
  const maxLoss = Math.max(...losses, 0.000001) * 1.08;
  let chart = frame(svgLoss, losses.length, 0, maxLoss, data.metadata.true_change_after, k);
  shape(svgLoss, 'polyline', {
    points: losses.map((y, i) => String(chart.px(i+1)) + ',' + String(chart.py(y))).join(' '),
    class: 'chart-loss-line'
  });
  const bounds = data.splits.map(item => item.bound);
  const maxBound = Math.max(1, ...bounds) * 1.05;
  chart = frame(svgBound, losses.length, 0, maxBound, data.metadata.true_change_after, k);
  shape(svgBound, 'polyline', {
    points: data.splits.map(item => String(chart.px(item.k)) + ',' + String(chart.py(item.bound))).join(' '),
    class: 'chart-bound-line'
  });
  data.splits.forEach(item => shape(svgBound, 'circle', {
    cx: chart.px(item.k), cy: chart.py(item.bound), r: item.k === k ? 6 : 3,
    class: item.k === k ? 'chart-bound-dot selected' : 'chart-bound-dot'
  }));
  const summary = document.getElementById('split-metrics');
  summary.replaceChildren();
  [
    ['Selected split', k],
    ['ε = |μ̂₁ − μ̂₂|', row.epsilon.toFixed(6)],
    ['κ = n₂ / t', row.kappa.toFixed(3)],
    ['Bernstein terms', row.term_before.toPrecision(4) + ' + ' +
      row.term_after.toPrecision(4) + ' = ' + row.bound.toPrecision(4)]
  ].forEach(([title, value]) => {
    const card = document.createElement('article');
    const caption = document.createElement('span');
    caption.textContent = title;
    const number = document.createElement('strong');
    number.textContent = value;
    card.append(caption, number);
    summary.appendChild(card);
  });
  document.querySelectorAll('[data-split-k]').forEach(tr => {
    tr.classList.toggle('is-selected', Number(tr.dataset.splitK) === k);
  });
}
chooser.addEventListener('change', render);
render();
document.getElementById('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'abcd-lab-evidence.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
