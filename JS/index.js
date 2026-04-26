const toggle = document.getElementById('mobile-menu');
const menu = document.querySelector('.navbar_menu');

toggle.addEventListener('click', () => {
  menu.classList.toggle('active');
  toggle.classList.toggle('active');
});

const slider = document.querySelector('.slider');
const items = document.querySelectorAll('.item');
const resumeBtn = document.getElementById('resume-btn');
const cardInfo = document.getElementById('card-info');
const cardTitle = document.getElementById('card-title');
const cardDesc = document.getElementById('card-desc');
const itemCount = 6;
const degPerItem = 360 / itemCount;
let isPaused = false;
let totalRotation = 0;
let lastTimestamp = null;
const speed = 360 / 20000;
let typewriterTimeout = null;

function typeWriter(element, text, speed = 30) {
  clearTimeout(typewriterTimeout);
  element.textContent = '';
  let i = 0;
  function type() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      typewriterTimeout = setTimeout(type, speed);
    }
  }
  type();
}

function tick(timestamp) {
  if (!isPaused) {
    if (lastTimestamp !== null) {
      totalRotation += (timestamp - lastTimestamp) * speed;
    }
    lastTimestamp = timestamp;
    requestAnimationFrame(tick);
  }
}

requestAnimationFrame(tick);

function getCurrentRotation() {
  const style = window.getComputedStyle(slider);
  const matrix = new DOMMatrix(style.transform);
  return Math.atan2(matrix.m13, matrix.m33) * (180 / Math.PI);
}

items.forEach((item) => {
  item.addEventListener('click', () => {
    const position = parseInt(item.style.getPropertyValue('--position'));
    const cardAngle = (position - 1) * degPerItem;
    const currentRot = getCurrentRotation();
    let delta = (-cardAngle - currentRot) % 360;
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    if (Math.abs(delta) < 5) return;
    const snapped = currentRot + delta;

    slider.style.animation = 'none';
    slider.style.transform = `rotateX(9deg) rotateY(${currentRot}deg)`;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        slider.classList.add('snapping');
        slider.style.transform = `rotateX(9deg) rotateY(${snapped}deg)`;
        totalRotation = snapped;
        isPaused = true;
        lastTimestamp = null;
        resumeBtn.classList.add('visible');

        cardTitle.textContent = item.dataset.title;
        typeWriter(cardDesc, item.dataset.desc);
        cardInfo.classList.add('visible');
      });
    });
  });
});

resumeBtn.addEventListener('click', () => {
  slider.classList.remove('snapping');
  slider.style.animation = `autoRun 20s linear infinite`;
  slider.style.animationDelay = `-${(((totalRotation % 360) + 360) % 360) / 360 * 20}s`;
  isPaused = false;
  lastTimestamp = null;
  resumeBtn.classList.remove('visible');
  cardInfo.classList.remove('visible');
  requestAnimationFrame(tick);
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