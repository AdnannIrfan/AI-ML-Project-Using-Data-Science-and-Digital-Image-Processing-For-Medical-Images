# MedicalAI Pro — Complete Setup & Run Guide

## Project Structure
```
MedicalAI_Pro/
├── app.py                  ← Flask web application (main entry point)
├── train_nlp_model.py      ← NLP model training script
├── train_image_model.py    ← Image model training script
├── requirements.txt        ← All Python dependencies
├── models/                 ← Saved trained models (created after training)
│   ├── nlp_model.pkl
│   └── image_model.pkl
├── charts/                 ← Generated charts (created after training)
├── uploads/                ← Uploaded images (created automatically)
└── templates/
    ├── base.html
    ├── home.html
    ├── upload.html
    ├── result.html
    ├── nlp.html
    ├── about.html
    └── contact.html
```

---

## STEP 1 — Download Kaggle Datasets

### Dataset 1 (for NLP model):
- URL: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
- Download: `mtsamples.csv`
- Place it in: `MedicalAI_Pro/mtsamples.csv`

### Dataset 2 (for Image model):
- URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Download and unzip so your folder looks like:
  ```
  MedicalAI_Pro/
  └── chest_xray/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/
          ├── NORMAL/
          └── PNEUMONIA/
  ```

> NOTE: If you don't have the datasets, the scripts run in fallback/demo mode automatically.

---

## STEP 2 — Open in VS Code

1. Open VS Code
2. File → Open Folder → select `MedicalAI_Pro`
3. Open Terminal: `` Ctrl+` `` (backtick) or Terminal → New Terminal

---

## STEP 3 — Create Virtual Environment

```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` at the start of your terminal prompt.

---

## STEP 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: Flask, scikit-learn, NLTK, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Pillow.

---

## STEP 5 — Train NLP Model

```bash
python train_nlp_model.py
```

This will:
- Load `mtsamples.csv` (or use demo mode)
- Preprocess text with NLTK
- Train TF-IDF + Ensemble (LR + SVM + RF)
- Run 5-fold cross-validation
- Save `models/nlp_model.pkl`
- Generate 6 charts in `charts/`

Training time: ~3–5 minutes

---

## STEP 6 — Train Image Model

```bash
python train_image_model.py
```

This will:
- Load chest_xray/ images (or use synthetic fallback)
- Extract 28 image features per X-ray
- Train RF-300 + GBM-150 + SVM-RBF ensemble
- Save `models/image_model.pkl`
- Generate evaluation charts in `charts/`

Training time: ~3–5 minutes

---

## STEP 7 — Run the Web App

```bash
python app.py
```

Open your browser: **http://127.0.0.1:5000**

---

## Features of the Web App

| Page | URL | Description |
|------|-----|-------------|
| Home | / | Overview, architecture |
| Image Analysis | /upload | Upload X-ray → predict Normal/Pneumonia |
| NLP Analysis | /nlp | Paste clinical text → classify specialty |
| About | /about | Model metrics, dataset links |
| Contact | /contact | Project info |

---



1. **Real Kaggle Datasets** — not synthetic/fake data
2. **Full NLP Pipeline** — tokenization, stopword removal, lemmatization, TF-IDF
3. **Complex Ensemble Models** — multiple algorithms combined with soft voting
4. **Cross-Validation** — 5-fold stratified CV for reliable evaluation
5. **28 Image Features** — statistics + texture + gradient (not just pixel values)
6. **Dual System** — both NLP AND image processing in one project
7. **Professional Web UI** — full Flask app with Bootstrap 5
8. **Complete Evaluation** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix, per-class F1
9. **8 Charts Generated** — EDA, preprocessing, training curves, evaluation

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `model not found` | Run training scripts first |
| `cv2 error` | `pip install opencv-python` |
| Port already in use | Change port: `app.run(port=5001)` |
| NLTK download fails | Run manually: `python -c "import nltk; nltk.download('all')"` |
