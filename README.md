# Brain Tumor MRI Classifier

ResNet50 Transfer Learning · Grad-CAM Explainability · 4-class MRI classification  
Student Capstone Project — not a medical device.

---

## Project Structure

```
project-root/
├── app/
│   └── streamlit_app.py      # Streamlit dashboard (main entry point)
├── models/
│   └── best_transfer_model.keras   # ← place model file here
├── notebooks/                # EDA, preprocessing, training, Grad-CAM analysis
├── outputs/                  # Plots, confusion matrices, heatmaps
├── src/                      # Reusable Python modules
├── requirements.txt
└── README.md
```

## Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the model file

Ensure `best_transfer_model.keras` is located at:

```
models/best_transfer_model.keras
```

relative to the project root (one level above the `app/` directory).

### 3. Launch the app

Run from the **project root**:

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## Model Details

| Property         | Value                                      |
|------------------|--------------------------------------------|
| Architecture     | ResNet50 (transfer learning)               |
| Strategy         | Frozen base + fine-tuned top 30 layers     |
| Input size       | 224 × 224 × 3 (raw 0–255 pixel values)     |
| Preprocessing    | Baked into model (ResNet50 preprocess_input) |
| Output classes   | glioma, meningioma, notumor, pituitary     |
| Test accuracy    | ~91–92%                                    |
| Explainability   | Grad-CAM (Keras-3-safe eager replay)       |

### Class ordering

Classes follow Keras `image_dataset_from_directory` alphabetical ordering:

| Index | Class      |
|-------|------------|
| 0     | glioma     |
| 1     | meningioma |
| 2     | notumor    |
| 3     | pituitary  |

---

## Known Limitations

- **Glioma / meningioma visual overlap** can cause misclassification, particularly
  at confidence scores below 80%.
- Trained on a single public benchmark dataset; real-world scanner diversity is
  not represented.
- Not validated on post-contrast, FLAIR, or non-T1 sequences.
- Grad-CAM provides a coarse saliency approximation — it is not a clinically
  validated lesion localisation method.

---

## Disclaimer

> This is a student capstone project for educational purposes only.
> It is **not** a clinically validated diagnostic tool.
> Predictions must **never** be used to inform real medical decisions.
> Always consult a qualified radiologist or neurologist.
