/* KL inspector for actual Stage-II candidate windows; no synthetic fallback or test decisions.
   Same event data exposed by detection_event_payload(). */
(() => {
  'use strict';
  document.addEventListener('DOMContentLoaded', () => {
    const root = document.querySelector('.eda-kl-inspector');
    const data = typeof edaData !== 'undefined'
      ? edaData.feature_chart?.cse_events?.kl_diagnostics : null;
    if (!root || !data || !data.events?.length) return;
    const eventSelect = document.getElementById('fbcsp-kl-event-select');
    const featureSelect = document.getElementById('fbcsp-kl-feature-select');
    if (!eventSelect || !featureSelect) return;
    const valid = data.events.findIndex(e => e.kl_status === 'diagnostic_only');
    if (valid >= 0) eventSelect.value = String(valid);

    const css = name => getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
    const teal = () => css('--eda-series-1') || '#54a6ff';
    const salmon = () => css('--eda-series-2') || '#ff7078';
    const muted = () => css('--muted') || '#94a3b8';
    const text = () => css('--text') || '#e2e8f0';

    function canvasArea(id, height = 300) {
      const el = document.getElementById(id);
      if (!el) return null;
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(320, el.getBoundingClientRect().width);
      el.width = Math.round(width * ratio);
      el.height = Math.round(height * ratio);
      el.style.height = height + 'px';
      const ctx = el.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      return {ctx, width, height};
    }

    function axes(ctx, w, h, yTitle, xTitle, top = 26) {
      const m = {left: 64, right: 20, top, bottom: 62};
      const width = w - m.left - m.right, height = h - m.top - m.bottom;
      ctx.fillStyle = css('--eda-plot-bg') || '#0d1728';
      ctx.fillRect(m.left, m.top, width, height);
      ctx.strokeStyle = css('--line') || '#64748b';
      ctx.beginPath(); ctx.moveTo(m.left, m.top);
      ctx.lineTo(m.left, m.top + height);
      ctx.lineTo(m.left + width, m.top + height); ctx.stroke();
      ctx.fillStyle = muted(); ctx.font = '12px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(xTitle, m.left + width / 2, h - 13);
      ctx.save(); ctx.translate(15, m.top + height / 2);
      ctx.rotate(-Math.PI / 2); ctx.fillText(yTitle, 0, 0); ctx.restore();
      return {...m, width, height};
    }

    function drawTimeline() {
      const area = canvasArea('fbcsp-kl-event-timeline', 255);
      if (!area) return;
      const {ctx, width, height} = area;
      const m = axes(ctx, width, height, 'Mean marginal KL', 'EWMA warning trial index', 62);
      const evaluated = data.events.filter(e => e.kl_status === 'diagnostic_only'
        && Number.isFinite(e.forward) && Number.isFinite(e.reverse));
      if (!evaluated.length) {
        ctx.fillStyle = muted(); ctx.textAlign = 'center';
        ctx.fillText('No eligible completed KL windows', m.left + m.width / 2, m.top + m.height / 2);
        return;
      }
      const all = data.events.map(e => e.alarm_index);
      const xMin = Math.min(...all), xMax = Math.max(...all);
      const yMax = Math.max(0.001, ...evaluated.flatMap(e => [e.forward, e.reverse])) * 1.12;
      const x = v => m.left + (v - xMin) / Math.max(1, xMax - xMin) * m.width;
      const y = v => m.top + m.height - (v / yMax) * m.height;
      ctx.font = '11px system-ui'; ctx.fillStyle = muted();
      ctx.textAlign = 'left'; ctx.fillText(String(xMin), m.left, m.top + m.height + 19);
      ctx.textAlign = 'right'; ctx.fillText(String(xMax), m.left + m.width, m.top + m.height + 19);
      ctx.textAlign = 'left'; ctx.fillText(yMax.toFixed(3), m.left + 5, m.top + 12);
      for (const [key, color, label] of [
        ['forward', teal(), 'Before || after'],
        ['reverse', salmon(), 'After || before'],
      ]) {
        ctx.strokeStyle = color; ctx.fillStyle = color;
        // Dots rather than a connected line: only selected warning windows are scored.
        evaluated.forEach(e => {
          ctx.beginPath(); ctx.arc(x(e.alarm_index), y(e[key]), 4, 0, Math.PI * 2); ctx.fill();
        });
        const offset = key === 'forward' ? 0 : 152;
        ctx.fillRect(m.left + 12 + offset, 28, 13, 4);
        ctx.fillStyle = text(); ctx.font = '11px system-ui';
        ctx.fillText(label, m.left + 30 + offset, 34);
      }
      const selected = data.events[Number(eventSelect.value)];
      if (selected?.kl_status === 'diagnostic_only') {
        ctx.strokeStyle = muted(); ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(x(selected.alarm_index), m.top + 25);
        ctx.lineTo(x(selected.alarm_index), m.top + m.height); ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    function drawComponents(e) {
      const area = canvasArea('fbcsp-kl-components', 280);
      if (!area) return;
      const {ctx, width, height} = area;
      const m = axes(ctx, width, height, 'Forward KL', 'Retained PCA component');
      if (e.kl_status !== 'diagnostic_only') return;
      const values = e.per_feature;
      const peak = Math.max(0.001, ...values) * 1.15;
      const step = m.width / values.length, bar = Math.max(5, step * 0.55);
      values.forEach((v, k) => {
        const barH = v / peak * m.height;
        ctx.fillStyle = teal();
        ctx.fillRect(m.left + (k + .5) * step - bar / 2,
          m.top + m.height - barH, bar, barH);
        ctx.fillStyle = muted(); ctx.textAlign = 'center'; ctx.font = '11px system-ui';
        ctx.fillText(data.feature_names[k] || 'PC'+(k+1),
          m.left + (k + .5) * step, m.top + m.height + 20);
        ctx.fillText(v.toFixed(3), m.left + (k + .5) * step,
          Math.max(m.top + 17, m.top + m.height - barH - 7));
      });
    }

    function drawProbabilities(e) {
      const area = canvasArea('fbcsp-kl-probabilities', 280);
      if (!area) return;
      const {ctx, width, height} = area;
      const m = axes(ctx, width, height, 'Probability / bin', 'Shared calibration histogram bin');
      if (e.kl_status !== 'diagnostic_only') return;
      const feature = Number(featureSelect.value);
      const hist = e.histograms[feature];
      if (!hist) return;
      const peak = Math.max(0.001, ...hist.reference, ...hist.current) * 1.15;
      const step = m.width / hist.reference.length, bar = Math.max(3, step * .3);
      hist.reference.forEach((v, k) => {
        const current = hist.current[k];
        const mid = m.left + (k + .5) * step;
        for (const [value, left, colour] of [
          [v, mid - bar - 1, teal()], [current, mid + 1, salmon()],
        ]) {
          const barH = value / peak * m.height;
          ctx.fillStyle = colour;
          ctx.fillRect(left, m.top + m.height - barH, bar, barH);
        }
        ctx.fillStyle = muted(); ctx.font = '10px system-ui'; ctx.textAlign = 'center';
        ctx.fillText(String(k+1), mid, m.top + m.height + 18);
      });
      ctx.textAlign = 'left'; ctx.font = '11px system-ui';
      ctx.fillStyle = teal(); ctx.fillRect(m.left + 8, m.top + 9, 12, 5);
      ctx.fillStyle = text(); ctx.fillText('Before', m.left + 25, m.top + 16);
      ctx.fillStyle = salmon(); ctx.fillRect(m.left + 98, m.top + 9, 12, 5);
      ctx.fillStyle = text(); ctx.fillText('After', m.left + 115, m.top + 16);
    }

    function render() {
      const e = data.events[Number(eventSelect.value)];
      if (!e) return;
      document.getElementById('fbcsp-kl-forward').textContent =
        Number.isFinite(e.forward) ? e.forward.toFixed(4) : '—';
      document.getElementById('fbcsp-kl-reverse').textContent =
        Number.isFinite(e.reverse) ? e.reverse.toFixed(4) : '—';
      document.getElementById('fbcsp-kl-hotelling').textContent = e.hotelling_status;
      const summary = document.getElementById('fbcsp-kl-event-summary');
      summary.textContent = e.kl_status === 'diagnostic_only'
        ? 'Warning trial '+e.alarm_index+' · Before '+e.reference_window.join('–')
          +' · After '+e.current_window.join('–')
          +' · KL available by trial '+e.validation_index
        : 'Warning trial '+e.alarm_index+' · KL not evaluated: '+e.kl_status
          +' (not a zero divergence)';
      drawTimeline(); drawComponents(e); drawProbabilities(e);
    }
    eventSelect.addEventListener('change', render);
    featureSelect.addEventListener('change', render);
    window.addEventListener('resize', () => {
      clearTimeout(window.__klInspectorResize);
      window.__klInspectorResize = setTimeout(render, 120);
    });
    render();
  });
})();
