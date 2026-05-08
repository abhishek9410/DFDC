# DeepFake_Detection
## Table of Contents:
- What is DeepFake?
- Demo of the Project
- Impact of DeepFake Videos
- Project Objectives
- Project Pipeline
  - Pre-processing WorkFlow
  - Prediction WorkFlow
- Models Usage and their Architecture
- Project Setup & Running
- Technologies Used
- Conclusion
- Team


## What is DeepFake?
- DeepFakes are images or videos which have been altered to feature the face of
someone else, like an advanced form of Face Swapping, using an AI DeepFake
Converter.
- Many Deep Fakes are done by superimposing or combining existing images into source
images and videos using Generative Adversarial Networks (GAN) and these networks
are developing better every day

## Demo of the Project
Link : https://www.youtube.com/watch?v=wy8mVnBZ6pY&ab_channel=BalajiKartheek

## Impact of DeepFake Videos
- DeepFakes can be used to create fake news, celebrity unusual videos, politician
content videos, and financial fraud.
- False Rumours can be spread using DeepFake videos which causes unrest and
mental anxiety among people.
- Many fields in Film Industry, content providers, and social media platforms are
fighting against DeepFake.
 
 # Project Objectives:
 
Identification of deepfakes is necessary to prevent the use of malicious AI.
We intend to,
-  Build a model that processes the given video and classifies it as REAL or FAKE.
-  Dploy a feature in the social media apps that can detect and give a warning to
the content provider who is willing to do viral by uploading deepFaked images or
videos.

![image](https://user-images.githubusercontent.com/77656115/206965843-6ac74168-3e31-43d6-9bbf-3e3d25e17522.png)

### Goal:
To Crate a deep learning model that is capable of recognizing deepfake images. A
thorough analysis of deepfake video frames to identify slight imperfections in the face
head and the model will learn what features differentiate a real image from a deepfake.

![image](https://user-images.githubusercontent.com/77656115/206965890-a1c345cf-8ae9-49f7-b498-ae4c7168666a.png)

### Project Pipeline

| Steps | Dscription |
| --- | --- |
| Step1 |    Loading the datasets |
| Step2 | Extracting videos from the dataset |
| Step3  | Extract all frames in the video for both real and fake |
| Step4 | Recognize the face subframe |
| Step5 |Locating the facial landmarks |
| Step6 |Frame-by-frame analysis to address any changes in the face landmarks |
| Step7 | To Classify the video either as REAL or Fake.|


## General WorkFlow:
### Pre-processing:
![image](https://user-images.githubusercontent.com/77656115/206968030-1e9729e7-8d34-4295-a110-d05ad0ade7bb.png)

### Prediction WorkFlow:
![image](https://user-images.githubusercontent.com/77656115/206968272-73db6238-79a0-46a1-ad5b-e651ad002322.png)

# Models Usage: 
### Models with CNN Architecture

Implemented the following models with CNN architecture
**MesoNet**
- This model is pre-trained to detect deepfake images, but it is bad at detecting Fake 
video frames
**ResNet50v**
- This model is trained using dee fake images cropped from the videos with preset 
weights of imagenet dataset
**EfficientNetB0**
- This model is also trained using deepfake images cropped from the videos with 
preset weights of imagenet dataset

### Models with CNN + Seqential Architecture
**InceptionV3(CNN Model) + GRU(sequential)**

-  This model works well because of both CNN and Sequential architecture.
- Test Accuracy is approx. 82%
- For Each Frame in the Video, it will generate the feature Vectors
- HyperParameters used: 
- Optimizer: Adam ( Adam Works fine as it changes the Learning Rate over time )
- Metric as Accuracy
- loss as sparse_categorical_crossentropy (loss function when there are two or more 
label classes )
- Among all the Optimizers Adam is Working Well.
- The accuracy of the model increases as the epochs are increasing.

**Limitations**
This model doesn’t work well when there are multiple faces in the Video, as it needs to 
detect the multiple faces in each Frame.

**EfficientNetB2(CNN Model) + GRU(sequential)**

- This model works well because of both CNN and Sequential architecture
- Test Accuracy is approx. 85%
- For Each Frame in the Video, it will generate the feature Vectors
- HyperParameters used: 
- Optimizer: Adam ( Adam Works fine as it changes the Learning Rate over time )
- Metric as Accuracy
- loss as sparse_categorical_crossentropy (loss function when there are two or more 
label classes )
- Among all the Optimizers Adam is Working Fine.
- The accuracy of the model increases as the epochs are increasing.
**Limitations**
- This model doesn’t work well when there is dark background in the video frames. As it is 
difficult to detect the faces in the Video Frame.

## Project Setup & Running

Follow these steps to set up and run the Flask app locally.

### 1. Prerequisites
- Python 3.8 or higher
- Windows, Linux, or macOS
- The trained classifier file at `Deploy/models/inceptionNet_model.h5`
- Optional: NVIDIA GPU with CUDA support for faster inference

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BalajiKartheek/DeepFake_Detection.git
   cd DeepFake_Detection
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   Linux/macOS:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```powershell
   cd Deploy
   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

### 3. Running the Application

1. **Start the Flask Server**:
   From the `Deploy` directory, run:
   ```powershell
   ..\.venv\Scripts\python.exe run_server.py
   ```

2. **Access the Web Interface**:
   Open your browser and navigate to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Verify the API Health Check**:
   In another terminal, run:
   ```powershell
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health
   ```

   A healthy server returns JSON with:
   ```json
   {
     "status": "ok",
     "models_loaded": true
   }
   ```

4. **Analyze a Video**:
   - Click **Choose File** to select an MP4, AVI, or MOV video.
   - Click **Upload**.
   - Once uploaded, click **Analyze** to see the prediction result (REAL or FAKE) and the confidence score.

### 4. Running in the Background on Windows

From the repository root:
```powershell
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "run_server.py" -WorkingDirectory ".\Deploy" -WindowStyle Hidden
```

To stop the server, find the Python process and stop it:
```powershell
Get-Process -Name python
Stop-Process -Id <PID>
```

*Note: The model (`inceptionNet_model.h5`) is located in `Deploy/models/`. Model loading may take several seconds on startup, and predictions can take about a minute for a 10-second 30fps video.*

## Accuracy Upgrade: Train the EfficientNetV2 Model

The deployed app now supports a stronger model path:

```text
Deploy/models/efficientnetv2_frame_model.keras
```

If this file exists, Flask uses it automatically. If it does not exist, Flask falls back to the original `InceptionV3 + GRU` model.

### 1. Extract the DFDC sample dataset

From the repository root:

```powershell
.\.venv\Scripts\python.exe "Model Training\train_frame_model.py" --extract-zip "Dataset\deepfake-detection-challenge\dfdc-train-sample-dataset.zip" --dataset-dir "Dataset\deepfake-detection-challenge" --max-videos 20 --epochs 1 --fine-tune-epochs 0
```

The command above is a smoke test. It verifies that extraction and training work, but it is not enough for high accuracy.

### 2. Train a real model

Use more videos and more epochs:

```powershell
.\.venv\Scripts\python.exe "Model Training\train_frame_model.py" --dataset-dir "Dataset\deepfake-detection-challenge" --frames-per-video 24 --epochs 8 --fine-tune-epochs 4 --batch-size 16
```

For better real-world accuracy, train on more than the small DFDC sample. Mix datasets such as DFDC, FaceForensics++, Celeb-DF v2, DeeperForensics, and WildDeepfake, and keep one dataset completely unseen for final testing.

### 3. Restart Flask

After training finishes, restart the server:

```powershell
Stop-Process -Id <SERVER_PID>
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "run_server.py" -WorkingDirectory ".\Deploy" -WindowStyle Hidden
```

Then check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health
```

The health response should show `EfficientNetV2 frame ensemble` as the active model.

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## Conclusion:

- In this project, we have implemented a method for the detection of Deep-Fake videos using the 
combination of CNN and RNN architecture. We have kept our focus on Face-Swapped Deep-Fake 
videos.

- We primarily experimented only with various pre-trained CNN models like EfficientNet, and 
ResNet by finding the probability of each video frame being fake and predicting the output based on an aggregate of these probabilities. But the results weren’t satisfactory, so we went forward by combining CNN and RNN models.

- For the CNN + RNN model, the features of face-cropped video frames are extracted using pretrained CNN models and it is passed onto the RNN model which classifies the video as REAL or 
FAKE. We Experimented with EfficientNet and inception net for the feature extraction part and 
GRU is used to make the classification. We have obtained a maximum Test Accuracy of ~85% 
using this approach. Our model has high precision for FAKE videos which is obtained by giving 
more FAKE videos during the training of the Model.


