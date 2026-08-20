(function () {
  "use strict";
  const payload = JSON.parse(document.getElementById("shift-data").textContent);
  const ns = "http://www.w3.org/2000/svg";

  function node(name, attributes) {
    const element = document.createElementNS(ns, name);
    Object.entries(attributes || {}).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function moments(points) {
    const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    let xx = 0, xy = 0, yy = 0;
    points.forEach((point) => {
      const dx = point.x - meanX, dy = point.y - meanY;
      xx += dx * dx; xy += dx * dy; yy += dy * dy;
    });
    const divisor = Math.max(points.length - 1, 1);
    xx /= divisor; xy /= divisor; yy /= divisor;
    const root = Math.sqrt(Math.max(0, ((xx - yy) / 2) ** 2 + xy ** 2));
    return {
      x: meanX, y: meanY,
      major: Math.sqrt(Math.max((xx + yy) / 2 + root, 0)),
      minor: Math.sqrt(Math.max((xx + yy) / 2 - root, 0)),
      angle: 0.5 * Math.atan2(2 * xy, xx - yy) * 180 / Math.PI
    };
  }

  function drawBoundary(svg, boundary, xScale, yScale, minX, maxX, cssClass) {
    if (!boundary || Math.abs(boundary.b) < 1e-9) return;
    const y1 = -(boundary.a * minX + boundary.c) / boundary.b;
    const y2 = -(boundary.a * maxX + boundary.c) / boundary.b;
    svg.appendChild(node("line", {x1: xScale(minX), y1: yScale(y1), x2: xScale(maxX), y2: yScale(y2), class: cssClass}));
  }

  function drawBand(container, band) {
    const width = 720, height = 470, margin = {top: 24, right: 24, bottom: 52, left: 62};
    const all = band.train.concat(band.test);
    let minX = Math.min(...all.map((point) => point.x)), maxX = Math.max(...all.map((point) => point.x));
    let minY = Math.min(...all.map((point) => point.y)), maxY = Math.max(...all.map((point) => point.y));
    const padX = Math.max((maxX - minX) * 0.18, 0.2), padY = Math.max((maxY - minY) * 0.18, 0.2);
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;
    const xScale = (value) => margin.left + (value - minX) / (maxX - minX) * (width - margin.left - margin.right);
    const yScale = (value) => height - margin.bottom - (value - minY) / (maxY - minY) * (height - margin.top - margin.bottom);
    const svg = node("svg", {viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true"});
    svg.appendChild(node("line", {x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis"}));
    svg.appendChild(node("line", {x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis"}));

    [[band.train, "train"], [band.test, "test"]].forEach(([points, split]) => {
      points.forEach((point) => svg.appendChild(node("circle", {cx: xScale(point.x), cy: yScale(point.y), r: 3.2, class: `point ${split}`})));
      const stats = moments(points);
      const scaleX = (width - margin.left - margin.right) / (maxX - minX);
      const scaleY = (height - margin.top - margin.bottom) / (maxY - minY);
      svg.appendChild(node("ellipse", {
        cx: xScale(stats.x), cy: yScale(stats.y), rx: 2 * stats.major * scaleX,
        ry: 2 * stats.minor * scaleY, transform: `rotate(${-stats.angle} ${xScale(stats.x)} ${yScale(stats.y)})`,
        class: `distribution ${split}`
      }));
    });
    drawBoundary(svg, band.boundaries.train, xScale, yScale, minX, maxX, "boundary train");
    drawBoundary(svg, band.boundaries.test, xScale, yScale, minX, maxX, "boundary test");
    const xLabel = node("text", {x: width / 2, y: height - 12, class: "axis-label"}); xLabel.textContent = "CSP feature 1"; svg.appendChild(xLabel);
    const yLabel = node("text", {x: 17, y: height / 2, class: "axis-label", transform: `rotate(-90 17 ${height / 2})`}); yLabel.textContent = "CSP feature 2"; svg.appendChild(yLabel);
    container.appendChild(svg);
  }

  document.querySelectorAll("[data-band]").forEach((container) => {
    drawBand(container, payload.bands.find((band) => band.id === container.dataset.band));
  });
})();
