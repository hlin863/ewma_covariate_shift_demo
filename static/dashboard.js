/* Shared theme controls used by the standalone Learning pages.
 * No server calls, no experiment execution; safe for GitHub Pages snapshots.
 */
(() => {
  "use strict";
  const root = document.documentElement;
  const key = "dashboard-theme";
  const buttons = Array.from(document.querySelectorAll("[data-theme-option]"));
  function applyTheme(theme) {
    if (theme !== "light" && theme !== "dark") return;
    root.dataset.theme = theme;
    try { localStorage.setItem(key, theme); } catch (_) {}
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(button.dataset.themeOption === theme));
    }
  }
  for (const button of buttons) {
    button.addEventListener("click", () => applyTheme(button.dataset.themeOption));
  }
  applyTheme(root.dataset.theme === "light" ? "light" : "dark");
})();
