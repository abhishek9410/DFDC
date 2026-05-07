

- DeepFake Detection Challenge Dataset consists of the large collection of video and audio altered files.
- A Sample of this Dataset is used for our project.

link :  https://www.kaggle.com/competitions/deepfake-detection-challenge/data

- Train Data: It consists of 400 videos in .mp4 format.
- Test Data: It consists of 400 videos in .mp4 format.

## Download locally

1. Accept the Kaggle competition rules in the browser.
2. Create a Kaggle API token from Account settings and save it as:
   `Dataset\.kaggle\kaggle.json`
3. Install the Kaggle CLI:
   `.\.venv\Scripts\python.exe -m pip install kaggle`
4. From the repository root, download and extract the sample data:
   `.\Dataset\download_dfdc_sample.ps1`
5. Train the deployed model:
   `.\.venv\Scripts\python.exe ".\Model Training\train_dfdc.py"`

The training script expects the Kaggle sample at
`Dataset\deepfake-detection-challenge\train_sample_videos`.

## MetaData of the Dataset
- filename - the filename of the video
- label - whether the video is REAL or FAKE
   -- y is 1 if the video is FAKE, 0 if REAL
- origianl - in the case that a train set video is FAKE, the original video is listed here
- split- this is always equal to “train”.
