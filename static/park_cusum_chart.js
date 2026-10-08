/* Lightweight self-contained SVG line plot: no external chart dependency. */
(() => {
  const target = document.getElementById('park-cusum-svg');
  const payload = document.getElementById('park-cusum-data');
  if (!target || !payload) return;
  const data = JSON.parse(payload.textContent);
  const xs = data.positions, maxValues = data.maximum, avgValues = data.average;
  if (!xs.length) return;
  const ns = 'http://www.w3.org/2000/svg';
  const chart = {x: 62, y: 18, w: 724, h: 205};
  const lo = xs[0], hi = xs[xs.length - 1], ceiling = Math.max(...maxValues, ...avgValues, 0.0001) * 1.08;
  const xx = v => chart.x + (v - lo) * chart.w / Math.max(hi-lo, 1);
  const yy = v => chart.y + chart.h - v / ceiling * chart.h;
  const el = (tag, attrs, value) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k,String(v)));
    if (value != null) node.textContent = value;
    target.appendChild(node);
    return node;
  };
  const colorMax = '#c97738', colorAvg = '#338baf';
  for (let tick=0; tick<=4; tick++) {
    const y = chart.y + tick * chart.h/4;
    el('line',{x1:chart.x,y1:y,x2:chart.x+chart.w,y2:y,stroke:'currentColor','stroke-opacity':0.13});
    el('text',{x:chart.x-10,y:y+4,'text-anchor':'end',fill:'currentColor','font-size':11}, (ceiling*(1-tick/4)).toFixed(2));
  }
  [lo,Math.round((lo+hi)/2),hi].forEach(t => el('text',{x:xx(t),y:246,fill:'currentColor','font-size':11,'text-anchor':'middle'},t));
  el('text',{x:chart.x+chart.w/2,y:264,fill:'currentColor','font-size':11,'text-anchor':'middle'},'Candidate split position (ordered observations)');
  [[maxValues,colorMax],[avgValues,colorAvg]].forEach(([values,color])=>{
    const points = values.map((v,i) => `${xx(xs[i]).toFixed(2)},${yy(v).toFixed(2)}`).join(' ');
    el('polyline',{points,fill:'none',stroke:color,'stroke-width':2.5,'stroke-linejoin':'round'});
  });
  [[data.b_max,colorMax],[data.b_avg,colorAvg]].forEach(([peak,color])=>{
    el('line',{x1:xx(peak),y1:chart.y,x2:xx(peak),y2:chart.y+chart.h,stroke:color,
      'stroke-width':1.5,'stroke-dasharray':'4 5','stroke-opacity':0.85});
  });
})();
