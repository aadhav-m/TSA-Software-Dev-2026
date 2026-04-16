const toggle = document.getElementById('mobile-menu');
const menu = document.querySelector('.navbar_menu');

toggle.addEventListener('click', () => {
  menu.classList.toggle('active');
  toggle.classList.toggle('active');
});

const videoElem = document.getElementById("video");
const stopButton = document.getElementById("stopButton");
const startButton = document.getElementById("startButton");

async function accessFeatures() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    videoElem.srcObject = stream;
    console.log("Webcam access granted");
  } catch (error) {
    console.log("Error while retrieving webcam permission", error)
  }
}

function stopFeatures() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    videoElem.srcObject = null;
    stream = null;
    console.log("Webcam has ended")
  }
}

startButton.addEventListener("click", accessFeatures);
stopButton.addEventListener("click", stopFeatures);

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