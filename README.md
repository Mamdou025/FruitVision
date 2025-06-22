# FruitVision

FruitVision is a small deep learning project for fruit recognition using TensorFlow and Streamlit.

## Installation

1. Create and activate a Python 3.11 virtual environment.
2. Install the trimmed set of requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) verify the setup with:
   ```bash
   python test_installation.py
   ```

## Create the unified dataset

The script `Edufruits_v3.py` can build a cleaned dataset from the original `data/fruits-360` directory and train the model.

Run it directly to perform both steps:
```bash
python Edufruits_v3.py
```
This creates the folder `data_v3/` with unified class folders and starts the training process. Model weights are saved inside `models/`.

If you only want to create the dataset without training, call `create_unified_dataset_v3()` from Python:
```bash
python - <<'PY'
from Edufruits_v3 import EduFruisV3Fixed
EduFruisV3Fixed().create_unified_dataset_v3()
PY
```

## Launch the Streamlit interface

After training, start the interactive demo with:
```bash
streamlit run Streamlit_app/fruivision_v3.py
```

## Legacy material

Older experiments and notebooks are stored under `archive/`.
