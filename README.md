# Aedes Mosquitoes Classification Using Deep Learning and Computer Vision Techniques
---
This repository contains the code of the project in a Jupyter Notebook and the necessary resources. The objective was to classify Aedes mosquitos between two particular species: *Aegypti* and *Albopictus*. The task will be addressed by constructing a Convolutional Neural Network (CNN) capable of predicting whether a mosquito image belongs to *Aegypti* or *Albopictus*.

# Directory Structure
---
* `notebook.ipynb`: Jupyter Notebook containing the project.
* `requirements.txt`: Requirements for the project.
* `util/`: Folder containing scripts required for the proper execution of the Notebook.
* `gui/`: Folder containing the GUI script used for testing the final model.

The final trained model is not inlcuded in this repository. You can access to it through the following link:

* [Aedes Classification Trained Model (Google Drive)](https://drive.google.com/drive/folders/1LvRJgP3TuDlwX755R4j6fICcbMMVHiCj?usp=sharing)

This Drive includes the model itself, summaries and plots.

# Requirements
---
This project requires Python 3.13.7 and the packages listed in `requirements.txt`. To install the necessary dependencies, you can run the following command:

```bash
# Install dependencies
pip install tensorflow==2.20.0 \
            numpy>=1.25.0 \
            pandas==2.3.2 \
            matplotlib==3.10.5 \
            seaborn>=0.12.2 \
            scikit-learn>=1.3.1 \
            Pillow>=10.0.0 \
            ipywidgets>=8.2.0 \
            h5py>=3.9.0 \
            protobuf>=4.24.0 \
            packaging>=23.3 \
            kaggle==1.7.4.5
```

Alternatively, you can install all dependencies directly from the requirements.txt file: `pip install -r requirements.txt`

Notes:
* Make sure you are using a compatible Python version (e.g., Python 3.11 or later).
* For Windows users using CMD, you may need to run the command in a single line or use PowerShell.
* Some packages, such as `tensorflow` or `ipywidgets`, may require additional setup for GPU support or Jupyter Notebook integration.

It is also recommended to use a **virtual environment** to manage dependencies. You can create and activate one as follows:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

Once the virtual environment is activated, install the required packages.

# Data Used
---
https://www.kaggle.com/datasets/nicolaspalomares/aedes-mosquitos-dataset-reduced
*Aedes Mosquitos Dataset (Reduced) - Aedes aegypti (Linnaeus) and Aedes albopictus (Skuse)*

Dataset reduced from original by *Pradeep Isawasan*: https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos
