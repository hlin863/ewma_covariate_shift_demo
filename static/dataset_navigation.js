document.querySelectorAll('.nav-menu').forEach(menu => {
  menu.addEventListener('toggle', () => {
    if (menu.open) {
      document.querySelectorAll('.nav-menu').forEach(other => {
        if (other !== menu) other.open = false;
      });
      const panel = menu.querySelector('.nav-menu-panel');
      const summary = menu.querySelector('summary').getBoundingClientRect();
      panel.style.position = 'fixed';
      panel.style.right = 'auto';
      panel.style.left = Math.max(12, Math.min(summary.left, window.innerWidth - panel.offsetWidth - 12)) + 'px';
      panel.style.top = (summary.bottom + 8) + 'px';
      panel.style.maxHeight = Math.max(100, window.innerHeight - summary.bottom - 20) + 'px';
      panel.style.overflowY = 'auto';
    }
  });
  menu.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      menu.open = false;
      menu.querySelector('summary').focus();
    }
  });
  document.addEventListener('click', event => {
    if (!menu.contains(event.target)) menu.open = false;
  });
});
['scroll', 'resize'].forEach(eventName => window.addEventListener(eventName, () => {
  document.querySelectorAll('.nav-menu[open]').forEach(menu => { menu.open = false; });
}));
