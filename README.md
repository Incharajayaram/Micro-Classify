
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

**Deployment**: Docker, gunicorn, nginx, AWS ec2, Route 53, certbot.<br>

**API Testing**: Postman API.<br>

**Version Control & CI/CD**: Git/GitHub.<br>

## Project Structure

```sh
Micro-Classify/
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

## Model Metrics 
```sh
Test Loss: 0.0244, Accuracy: 0.9918
Classification Report:
                      precision    recall  f1-score   support

  3_long_blade_rotor       0.99      0.99      0.99        72
 3_short_blade_rotor       0.99      0.96      0.98        85
                Bird       1.00      1.00      1.00        76
Bird+mini-helicopter       1.00      1.00      1.00        78
               drone       1.00      1.00      1.00        85
            rc_plane       0.98      1.00      0.99        90

            accuracy                           0.99       486
           macro avg       0.99      0.99      0.99       486
        weighted avg       0.99      0.99      0.99       486
```
## Graphs 
**Loss Curve**<br>
![image](https://github.com/user-attachments/assets/c472c727-1e3a-4679-9f45-cd56a2f74178)
<br>**Accuracy Curve**<br>
![image](https://github.com/user-attachments/assets/8e974691-d29f-4391-9fb7-9cce5fdcd759)
<br>**F1 score Curve**<br>
![image](https://github.com/user-attachments/assets/a06316a7-d2b1-4adb-b2b6-8936e2684846)

## Web App
![image](https://github.com/user-attachments/assets/97f0194d-bf9a-4bce-91a3-cb495ae69664)

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
