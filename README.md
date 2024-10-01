
# Micro-Classify
Team Name: RadarVision <br>
ID: 39446<br>

[Web App Repository](https://github.com/Incharajayaram/micro-doppler-web-app)<br>
<br>[Prototype Repository](https://github.com/Incharajayaram/streamlit-app-microclassify)</br>








 



## Contributors 

**[Inchara J](https://github.com/Incharajayaram)**<br>

**[Shravya H Jain](https://github.com/shravya312)**<br>

**[Diptangshu Bej](https://github.com/DiptangshuBej)**<br>

**[Anand Raut](https://github.com/Anand-Raut9)**<br>

**[Likith Manjunatha](https://github.com/Likith-m-22)**<br>

**[Chethan A C](https://github.com/chethanac15)**<br>

## Tech Stack

**Frontend**: HTML, CSS, JavaScript.<br>

**Backend**: Flask.<br>

**Machine Learning**: Python, PyTorch, Scikit-learn, numpy.<br>

**Visualization**: Matplotlib.<br>

**Deployment**: Docker, gunicorn, nginx, Route 53, certbot.<br>

**API Testing**: Postman API.<br>

**Version Control & CI/CD**: Git/GitHub.<br>

## Project Structure

```sh
Micro-Doppler-Based-Target-Classification-/
├── Backend/                       # Flask backend directory
│   ├── static/                    # Static files like css, js, images, etc
│   ├── templates/                 # Frontend html files as templates for Flask
│   ├── app.py                     # Main Flask App
│   ├── venv                       # virtual env Folders
│   ├── .gitignore
│   ├── monitor.py                 # file to monitor errors and logs
│   ├── requirements.txt           # flask requirements
│   ├── .env                       # General environment variables
│   └── README.md                  # Backend documentation
├── ml_model/                      # Machine learning model directory
│   ├── notebooks/                 # Jupyter notebooks for experiments and model training
│   ├── src/
│   │   ├── data/                  # Data handling scripts
│   │   ├── model/                 # Model training, evaluation, and prediction scripts
│   │   ├── utils/                 # Utility scripts (data preprocessing, visualization)
│   │   ├── main.py                # Main script python file to run the pretrained moddel
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
