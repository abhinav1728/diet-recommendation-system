document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  const weight = document.querySelector('input[name="weight"]');
  const height = document.querySelector('input[name="height"]');
  const age = document.querySelector('input[name="age"]');
  const button = document.querySelector('button[type="submit"]');

  form.addEventListener("submit", function (e) {
    if (
      weight.value < 30 || weight.value > 200 ||
      height.value < 100 || height.value > 250 ||
      age.value < 10 || age.value > 100
    ) {
      e.preventDefault();
      alert("⚠️ Please enter realistic values:\n• Weight: 30–200 kg\n• Height: 100–250 cm\n• Age: 10–100 years");
      return;
    }
    button.disabled = true;
    button.innerText = "Loading...";
    button.style.opacity = 0.7;
  });
});
