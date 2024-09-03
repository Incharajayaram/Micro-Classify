document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const form = event.target;
    const formData = new FormData(form);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').innerText = `Prediction: ${data.prediction}`;
        document.getElementById('confidence').innerText = `Confidence: ${data.confidence.toFixed(4)}`;
    })
    .catch(error => console.error('Error:', error));
});
