const toggle = document.getElementById('mobile-menu');
const menu = document.querySelector('.navbar_menu');

toggle.addEventListener('click', () => {
  menu.classList.toggle('active');
  toggle.classList.toggle('active');
});

const buttons = document.querySelectorAll('.button-neon');

buttons.forEach(btn => {
  const duration = 3.5 + Math.random() * 1.5;
  btn.style.setProperty('--pulse-speed', `${duration}s`);
  function randomize() {
    const hue = Math.floor(Math.random() * 260 + 100);
    btn.style.setProperty('--hue', hue);

    const nextDelay = 50 + Math.random() * 200;
    setTimeout(randomize, nextDelay);
  }

  setTimeout(randomize, Math.random() * 500);
});