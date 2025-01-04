
# Micro-Classify
Team Name: RadarVision <br>
ID: 39446<br>

[Web App Repository](https://github.com/Incharajayaram/radarvision-webpage)<br>


## Introduction 
The **Micro-Classify** project leverages **Micro-Doppler** radar signatures to identify and classify small aerial targets like **drones** and **birds**. This classification system is essential for surveillance, defence, and airspace management, providing a reliable distinction between human-made and natural objects.


## Problem Statement 

Current radar systems often struggle to differentiate between small aerial targets like drones and birds due to similarities in their radar signatures. Misclassification can lead to operational inefficiencies, such as false alarms or overlooked threats.

### The primary challenges include:

  1.	Variability in micro-Doppler signatures across radar bands.
  2. Imbalanced and limited datasets.
  3.	Achieving real-time, high-accuracy predictions.


## Solution Overview 

The proposed solution leverages a CNN-LSTM architecture with attention mechanisms to extract spatial and temporal features from micro-Doppler radar signals. Data augmentation via ACGANs (Auxiliary Classifier GANs) enhances dataset size and balance, improving classification performance.


## Dataset

  1. **Source**: Radio Micro-Doppler signatures
  2. **Size** [Datasetsize] with [number] samples
  3. **Classes**: Drones, birds and other Ariel targets
  4. **Preprocessing**:
        - Spectrogram generation
        - Noise reduction and normalization
  5. **Challenges**:
        - Class imbalance resolved using ACGANs.
        - High computational resource requirements for real time processing


## Deployment

  - **Platform**: Hosted on AWS EC2.
  - **Containerization**: Docker ensures scalability and reproducibility.
  - **API Access**: Interactive API built using Flask.
  - **Raspberry PI 5**: Built a data pipeline to stream spectrograms and showcase results on raspberry pi 5, by deploying the ml model on it.


## Evaluation Metrics

  - **Accuracy**: Measures overall classification correctness. 
  - **F1-Score**: Balances precision and recall for imbalanced datasets.
  - **Confusion Matrix**: Visualizes classification performance across classes.
  - **Latency**: Evaluates real-time prediction feasibility.


## Results 

- **Accuracy: 99.18%**
- **F1-Score: 0.99**
- **Latency: 12.8 ms**
- **Architecture**: ![image](https://github.com/user-attachments/assets/e298b10f-64a8-4fc4-9db0-c553ad1a9f4a)

- **ROC curve**: ![image](https://github.com/user-attachments/assets/46c27615-86b7-4d1c-bf58-fb93439ea886)

- **Confusion Matrix**: ![image](https://github.com/user-attachments/assets/79e61f3c-e61b-4ec8-8a6b-97d7a8c5130e)



## References

- Research papers on micro-Doppler radar classification.
- Pytorch and Scikit-Learn official documentation.



 



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
│   │   ├── main.py                # Main script python file to run the pre-trained model
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
![image](https://github.com/user-attachments/assets/7279b6a4-0468-4c2e-9a0f-8f22806b5c76)


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
