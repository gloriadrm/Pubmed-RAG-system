function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${target}`;
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
});
