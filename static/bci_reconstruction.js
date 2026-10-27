// Read saved evidence only; plotting does not rerun or transform model predictions.
(() => {
  const source = document.getElementById("bci-trace-data");
  const svg = document.getElementById("bci-loss-chart");
  if (!source || !svg) return;
  const rows = JSON.parse(source.textContent).filter(r => Number.isFinite(Number(r.reconstruction_mse)));
  if (!rows.length) return;
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag, attrs) => { const el = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,String(v))); svg.appendChild(el); return el; };
  const left=55,right=935,top=18,bottom=228;
  const ymax=Math.max(...rows.map(r=>Number(r.reconstruction_mse)),1e-12)*1.08;
  const xmax=Math.max(rows.length-1,1);
  add("path",{d:`M${left} ${top}V${bottom}H${right}`,stroke:"currentColor","stroke-opacity":"0.55",fill:"none"});
  const points=rows.map((r,i)=>`${left+i/xmax*(right-left)},${bottom-Number(r.reconstruction_mse)/ymax*(bottom-top)}`).join(" ");
  add("polyline",{points,fill:"none",stroke:"#239b8f","stroke-width":"2"});
  add("text",{x:left,y:248,fill:"currentColor","font-size":"12"}).textContent="Evaluation trial index →";
  add("text",{x:left+4,y:14,fill:"currentColor","font-size":"12"}).textContent=`MSE (max displayed: ${ymax.toPrecision(3)})`;
})();
