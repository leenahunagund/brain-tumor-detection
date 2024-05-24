
<p align="center"><img src="https://raw.githubusercontent.com/Oct4Pie/brain-tumor-detection/main/logo/brain.png" width="10%" style="margin: 0px; transform: translateY(-50%)">
</p>

# Brain-tumor-detection (CNN)
<img src="https://i.imgur.com/C0rTivW.png">

## About
This program is designed to facilitate the diagnosis of brain tumors from 2D MRI scan images, ensuring both accuracy and timeliness. The model is created using the TensorFlow API in Python, leveraging the high-level Keras API. The image classifier is based on a deep Convolutional Neural Network (CNN). For more information, try the [Streamlit app](https://brain-tumor-detection0.streamlit.app).

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Oct4Pie/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Create and activate a virtual environment (Python 3.9+):**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

## Usage
### Streamlit Web App
- Upload an MRI image via the Streamlit interface.
- The app will automatically crop the image to the brain area, analyze it, and display the results, including whether a tumor is detected and the confidence of the prediction.

### Running the Model
- The model can be trained, evaluated, and used for predictions using the provided Python scripts. Make sure to adjust the file paths and parameters as needed.

## File Structure
```
brain-tumor-detection/
│
├── app.py                    # Main entry point for the Streamlit app
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
├── logo/
├── model/
│   ├── __pycache__/
│   ├── archive3/
│   │   ├── no/
│   │   └── yes/
│   ├── brain_tumor_dataset/
│   │   ├── no/
│   │   └── yes/
│   ├── cropped/
│   │   ├── no/
│   │   └── yes/
│   ├── history/
│   ├── logs/
│   │   └── 20240523-154553/        # Tensorboard log
│   │       ├── train/
│   │       └── validation/
│   ├── models/         # Trained models
│   └── tests/
├── pages/
│   ├── __pycache__/
│   ├── _pages/
│   │   └── __pycache__/
│   ├── components/
│   ├── css/
│   └── samples/
└── utils.py
```

## Acknowledgements
- [Brain Tumor Classification (MRI)](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) (datasets)
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) (datasets)
- [MRI Based Brain Tumor Images](https://www.kaggle.com/mhantor/mri-based-brain-tumor-images) (datasets)
- [Starter: Brain MRI Images for Brain](https://www.kaggle.com/kerneler/starter-brain-mri-images-for-brain-b5be8b94-c) (datasets and inspiration)
