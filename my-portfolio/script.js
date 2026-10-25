const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector("#main-nav");
if (menuButton && nav) {
  menuButton.addEventListener("click", () => {
    const isOpen = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-expanded", String(!isOpen));
    menuButton.querySelector("span").textContent = isOpen ? "+" : "−";
    nav.classList.toggle("is-open", !isOpen);
  });
  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuButton.setAttribute("aria-expanded", "false");
      menuButton.querySelector("span").textContent = "+";
      nav.classList.remove("is-open");
    });
  });
}
const year = document.querySelector("#year");
if (year) year.textContent = String(new Date().getFullYear());
