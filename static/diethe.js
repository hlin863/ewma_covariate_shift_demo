'use strict';
const experiment = JSON.parse(document.getElementById('experiment-data').textContent);
const select = document.getElementById('policy');
const svg = document.getElementById('accuracy-chart');
const ns = 'http://www.w3.org/2000/svg';
function element(tag, attributes, text) {
  const node = document.createElementNS(ns, tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  svg.appendChild(node);
  return node;
}
function render() {
  const run = experiment.runs[select.value];
  svg.replaceChildren();
  const n = run.trials.length, x = i => 55 + i / Math.max(1, n - 1) * 875;
  [0, 0.5, 1].forEach(value => {
    const y = 220 - value * 185;
    element('line', {x1:55,x2:930,y1:y,y2:y,stroke:'#d7e3dd'});
    element('text', {x:8,y:y+5,fill:'#344e44','font-size':14}, `${value*100}%`);
  });
  element('text',{x:55,y:249,'font-size':14},'Trial 0');
  element('text',{x:860,y:249,'font-size':14},`Trial ${n-1}`);
  if (experiment.metadata.change_index !== null) {
    const at = x(experiment.metadata.change_index);
    element('line',{x1:at,x2:at,y1:20,y2:220,stroke:'#6b7570','stroke-dasharray':'6 4'});
  }
  const points = run.trials.filter(t => t.rolling_accuracy !== null)
    .map(t => `${x(t.trial_index)},${220-185*t.rolling_accuracy}`).join(' ');
  element('polyline',{points,fill:'none',stroke:'#176da0','stroke-width':2.5});
  run.updates.forEach(e => element('line',{x1:x(e.trigger_trial_index),x2:x(e.trigger_trial_index),y1:207,y2:220,stroke:'#b66514','stroke-width':3}));
  const body = document.getElementById('event-rows');body.replaceChildren();
  run.updates.forEach(e => {
    const row = document.createElement('tr');
    [e.trigger_trial_index,e.trigger_time,e.added_samples,e.training_size_after,e.classifier_version,(1000*e.retrain_seconds).toFixed(2)].forEach(v => {
      const cell=document.createElement('td');cell.textContent=v;row.appendChild(cell);
    });body.appendChild(row);
  });
  if (!run.updates.length) {const row=document.createElement('tr');const cell=document.createElement('td');cell.colSpan=6;cell.textContent='No updates for this policy.';row.appendChild(cell);body.appendChild(row);}
}
select.addEventListener('change',render);render();
document.getElementById('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(experiment,null,2)],{type:'application/json'}));
  const link = document.createElement('a');link.href=url;link.download='diethe-experiment.json';link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});
