
# Micro-Doppler-Based-Target-Classification

## Project Overview

The **Micro-Doppler Based Target Classification** project focuses on distinguishing between different Small Aerial Targets like Drones, RC Planes, Birds, etc by analyzing micro-Doppler signatures captured from radar sensors. <br>
This technology is essential for improving situational awareness in scenarios such as surveillance, search and rescue missions, and safeguarding critical infrastructure. <br>

The project encompasses:

**Signal Processing**: Obtained dataset from IEEE Dataport, which is around 4000 spectrogram images extracted from radar data. This data is normalized, and augmented. Feature extraction is used to to extract meaningful features                           from the spectrograms, to enhance model's performance.<br>
**Machine Learning**: Implementing Python-based models to classify objects based on the extracted features.<br>
**Web Application**: Creating an intuitive interface for users to interact with the classification system.<br>

## Contributors 

**[Inchara J](https://github.com/Incharajayaram)**<br>

**[Shravya H Jain](https://github.com/shravya312)**<br>

**[Diptangshu Bej](https://github.com/DiptangshuBej)**<br>

**[Anand Raut](https://github.com/Anand-Raut9)**<br>

**[Likith Manjunatha](https://github.com/Likith-m-22)**<br>

## Tech Stack

**Frontend**: React, HTML, Bootstrap, JavaScript.<br>

**Backend**: Flask/Express.js.<br>

**Machine Learning**: Python, PyTorch, Scikit-learn, numpy.<br>

**Visualization**: Matplotlib.<br>

**Deployment**: Docker, Gunicorn.<br>

**Version Control & CI/CD**: Git/GitHub, GitHub Actions.<br>

## Project Structure

```sh
Micro-Doppler-Based-Target-Classification-/
├── Frontend/                      # React frontend directory
│   ├── public/                    # Public assets like HTML, icons, etc.
│   ├── src/
│   │   ├── assets/                # Static assets (images, fonts, etc.)
│   │   ├── components/            # Reusable React components
│   │   ├── pages/                 # Page components (Home, Dashboard, etc.)
│   │   ├── services/              # API calls to Flask backend
│   │   ├── App.js                 # Main React component
│   │   ├── index.js               # React entry point
│   │   └── styles/                # CSS/Sass files
│   ├── .env                       # Environment variables
│   ├── package.json               # NPM dependencies
│   └── README.md                  # Frontend documentation
├── Backend/                       # Flask backend directory
│   ├── app/
│   │   ├── __init__.py            # Flask app initialization
│   │   ├── controllers/           # Route controllers
│   │   │   ├── classification.py  # Classification API endpoint
│   │   │   ├── healthcheck.py     # Healthcheck API endpoint
│   │   ├── models/                # Data models (if needed)
│   │   ├── services/              # Business logic (model inference, etc.)
│   │   │   ├── model_inference.py # Logic for loading models and making predictions
│   │   ├── utils/                 # Utility functions (e.g., logging, error handling)
│   │   ├── static/                # Static files (model weights, logs)
│   │   ├── templates/             # HTML templates (if needed for serving static pages)
│   │   └── config.py              # Configuration file (for environment variables, etc.)
│   ├── tests/                     # Unit tests for backend
│   │   ├── test_classification.py # Tests for the classification endpoint
│   │   └── conftest.py            # Test configurations and fixtures
│   ├── requirements.txt           # Python dependencies
│   ├── wsgi.py                    # WSGI entry point for production (optional)
│   ├── .flaskenv                  # Flask-specific environment variables
│   ├── .env                       # General environment variables
│   └── README.md                  # Backend documentation
├── ml_model/                      # Machine learning model directory
│   ├── notebooks/                 # Jupyter notebooks for experiments and model training
│   ├── src/
│   │   ├── data/                  # Data handling scripts
│   │   ├── model/                 # Model training, evaluation, and prediction scripts
│   │   ├── utils/                 # Utility scripts (data preprocessing, visualization)
│   ├── requirements.txt           # Python dependencies for ML
│   ├── venv/                      # Virtual environment for ML
│   ├── .gitignore                 # Ignore unnecessary files (e.g., model weights, virtual env)
│   └── README.md                  # ML model documentation
├── docker-compose.yaml            # Docker Compose file (if containerizing)
├── Dockerfile                     # Dockerfile for backend (if containerizing)
├── .gitignore                     # Global .gitignore file
├── LICENSE                        # License file
└── README.md                      # Main project documentation
```

## Contributing

1. **Fork the repository**
2. **Create a new branch**:

   ```sh
   git checkout -b feature
   ```

3. **Make your changes**
4. **Add your changes**:-

   ```sh
   git add <filename>
   ```
5. **Commit your changes**:

   ```sh
   git commit -m 'Add new feature'
   ```

6. **Push to the branch**:

   ```sh
   git push origin feature
   ```

7. **Create a new Pull Request**
