
# 🧠 SHL Grammar Scoring Engine

This project implements an automated grammar scoring system for spoken responses using the **Wav2Vec2** model. It processes `.wav` audio files and predicts grammar scores based on SHL assessment data.

---

## 📁 Folder Structure

```
project/
├── audio/
│   ├── train/
│   └── test/
├── train.csv
├── test.csv
├── sample_submission.csv (optional)
├── X_train.npy
├── y_train.npy
├── X_test.npy (optional)
└── submission.csv
```

---

## ⚙️ How It Works

### 1. **Feature Extraction**
- Uses `facebook/wav2vec2-base-960h` to convert audio files into dense feature embeddings.
- Features are extracted for both training and testing audio files.
- Feature extraction can be time-consuming (up to 6 hours), so we save them as `.npy` files for reuse.

### 2. **Model Training**
- Encodes labels using `LabelEncoder`.
- Balances the dataset to prevent class imbalance.
- Trains a **Random Forest Classifier** or **Ridge Regression** model.

### 3. **Evaluation**
- Splits the training data into train/validation sets.
- Evaluates with metrics like:
  - RMSE (Root Mean Squared Error)
  - Pearson Correlation
  - Classification Report
  - Confusion Matrix

### 4. **Test Prediction**
- Loads test audio and generates predictions.
- Converts predictions back to original score labels.
- Saves final predictions to `submission.csv`.

---

## 📦 Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages include:
- `transformers`
- `torch`, `torchaudio`
- `librosa`
- `scikit-learn`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `tqdm`

---

## 🚀 How to Run

### Step 1: Extract Features

```python
# Run once to extract and save train features
X_train = [...]  # Extracted features
y_train = [...]  # Corresponding labels

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
```

### Step 2: Load Saved Features & Train

```python
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Train model and evaluate
...
```

### Step 3: Predict on Test Data

```python
# Extract features for test files (can also cache with np.save)
...
# Predict and generate submission.csv
...
```

---

## 💡 Tips

- Make sure all `.wav` files listed in `train.csv` and `test.csv` are present in the corresponding folders.
- Use `librosa.load(path, sr=16000)` to ensure sampling rate compatibility with Wav2Vec2.
- Run heavy steps like feature extraction only once and cache with `.npy`.

---

- This model is tailored for SHL's spoken grammar scoring assessment.
- You may switch to regression models if predicting continuous scores.
- Use validation plots (confusion matrix, histograms) to better interpret performance.


