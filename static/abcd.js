'use strict';
const data = JSON.parse(document.getElementById('abcd-data').textContent);
const ns = 'http://www.w3.org/2000/svg';
const svgLoss = document.getElementById('loss-chart');
const svgBound = document.getElementById('bound-chart');
const chooser = document.getElementById('inspect-k');

function shape(root, kind, props, content) {
  const node = document.createElementNS(ns, kind);
  for (const [key, value] of Object.entries(props)) node.setAttribute(key, String(value));
  if (content !== undefined) node.textContent = String(content);
  root.appendChild(node);
  return node;
}
function frame(root, count, min, max, trueChange, selected) {
  root.replaceChildren();
  const px = i => 54 + i / count * 856;
  const py = v => 207 - 175 * (v - min) / Math.max(1e-15, max - min);
  for (let i = 0; i <= 4; i++) {
    const value = min + (max - min) * i / 4, y = py(value);
    shape(root, 'line', {x1:54,x2:910,y1:y,y2:y,stroke:'#d9e5df'});
    shape(root, 'text', {x:4,y:y+5,'font-size':12,fill:'#47655c'}, value.toPrecision(2));
  }
  shape(root, 'line', {x1:px(trueChange),x2:px(trueChange),y1:20,y2:207,stroke:'#6f7b79','stroke-dasharray':'5 5','stroke-width':2});
  shape(root, 'line', {x1:px(selected),x2:px(selected),y1:20,y2:207,stroke:'#bd7926','stroke-width':2});
  shape(root, 'text', {x:54,y:235,'font-size':13,fill:'#526660'}, '0');
  shape(root, 'text', {x:884,y:235,'font-size':13,fill:'#526660'}, String(count));
  shape(root, 'text', {x:730,y:19,'font-size':12,fill:'#526660'}, 'Gray = true; orange = selected');
  return {px, py};
}
function render() {
  const k = Number(chooser.value);
  const row = data.splits.find(s => s.k === k);
  const losses = data.observations.map(o => o.loss);
  const maxLoss = Math.max(...losses) * 1.08;
  let v = frame(svgLoss, losses.length, 0, maxLoss, data.metadata.true_change_after, k);
  shape(svgLoss, 'polyline', {
    points: losses.map((y,i) => v.px(i+1) + ',' + v.py(y)).join(' '),
    fill:'none',stroke:'#0d867b','stroke-width':2
  });
  const bounds = data.splits.map(s => s.bound), maxB = Math.max(1,...bounds)*1.05;
  v = frame(svgBound, losses.length, 0, maxB, data.metadata.true_change_after, k);
  shape(svgBound, 'polyline', {
    points: data.splits.map(s => v.px(s.k) + ',' + v.py(s.bound)).join(' '),
    fill:'none',stroke:'#3e678c','stroke-width':2.5
  });
  data.splits.forEach(s => shape(svgBound,'circle',{
    cx:v.px(s.k),cy:v.py(s.bound),r:s.k===k?6:3,
    fill:s.k===k?'#bd7926':'#3e678c'
  }));
  const box = document.getElementById('split-metrics');box.replaceChildren();
  const metrics = [
    ['Split',k],['ε = |μ̂₁−μ̂₂|',row.epsilon.toFixed(6)],
    ['κ = n₂/t',row.kappa.toFixed(3)],
    ['Term 1 + Term 2',row.term_before.toPrecision(4)+' + '+row.term_after.toPrecision(4)+' = '+row.bound.toPrecision(4)]
  ];
  metrics.forEach(([title,value]) => {
    const card=document.createElement('article');
    const lbl=document.createElement('span');lbl.textContent=title;
    const strong=document.createElement('strong');strong.textContent=value;
    strong.style.fontSize='1.1rem';card.append(lbl,strong);box.appendChild(card);
  });
}
chooser.addEventListener('change',render);render();
document.getElementById('download').addEventListener('click', () => {
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const link=document.createElement('a');
  link.href=url;link.download='abcd-lab-evidence.json';link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});
