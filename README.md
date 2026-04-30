# 📊 Perbandingan Machine Learning Konvensional vs Deep Learning

**Nama:** Muhammad Yoga Isnaeni  
**NIM:** 1202220266  
**Kelas:** SI-46-EDM  
**Universitas Telkom — Semester Genap 2025/2026**

---

## 📁 Struktur Repository

```
├── Kasus1.ipynb          # Titanic — Data Tabular (Random Forest, XGBoost, PyTorch MLP)
├── Kasus2.ipynb          # Digit Recognizer — Data Image (HOG+SVM vs Deep CNN)
├── Kasus3.ipynb          # Disaster Tweets — Data Teks (TF-IDF vs LSTM)
└── README.md
```

---

## ⚙️ Requirements & Instalasi

### Prasyarat Umum
- **Python** 3.8+
- **Google Colab** (direkomendasikan) atau environment lokal dengan GPU
- **Akun Kaggle** dengan API key aktif

### Install Dependensi

```bash
# Kasus 1 — Titanic
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch kagglehub

# Kasus 2 — Digit Recognizer
pip install pandas numpy matplotlib seaborn scikit-learn scikit-image tensorflow kagglehub

# Kasus 3 — Disaster Tweets
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow kagglehub
```

Atau install semua sekaligus:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch tensorflow scikit-image kagglehub
```

### Konfigurasi Kaggle API

Setiap notebook menggunakan `kagglehub` untuk download dataset otomatis. Pastikan kamu sudah set environment variable Kaggle:

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'
```

Cara mendapatkan API key: masuk ke **kaggle.com → Account → Create New API Token** → download `kaggle.json`.

---

## 🚢 Kasus 1 — Titanic (Data Tabular)

**File:** `Kasus1.ipynb`  
**Dataset:** [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
**Task:** Binary classification — memprediksi apakah penumpang selamat atau tidak

### Cara Menjalankan

1. Buka `Kasus1.ipynb` di Google Colab atau Jupyter
2. Jalankan cell instalasi di bagian atas:
   ```python
   !pip install xgboost
   !pip install kagglehub
   ```
3. Set Kaggle credentials (cell ke-2)
4. **Jalankan semua cell secara berurutan dari atas ke bawah** — ini penting karena data preprocessing di cell tengah bergantung pada variabel dari cell sebelumnya

### Alur Notebook

| Tahap | Deskripsi |
|---|---|
| **Setup & Import** | Install library, set Kaggle key, download dataset otomatis |
| **EDA** | Visualisasi survival rate berdasarkan Pclass dan Sex |
| **Preprocessing** | Cek missing values, imputasi Age/Embarked/Fare |
| **Feature Engineering** | Ekstrak Title dari Name, buat FamilySize, IsAlone, binning Age & Fare |
| **Encoding & Scaling** | Label encoding Sex, one-hot encoding kategorikal, StandardScaler |
| **Train-Val Split** | 80:20 split dengan `random_state=42` |
| **Model Konvensional** | GridSearchCV untuk Random Forest dan XGBoost (5-fold CV) |
| **Model Deep Learning** | PyTorch MLP — 2 hidden layer dengan Dropout + EarlyStopping |
| **Evaluasi & Perbandingan** | Accuracy, precision, recall, F1-score + leaderboard |
| **Error Analysis** | Analisis 32 prediksi salah XGBoost pada data validasi |

### Hasil

| Model | Accuracy | F1-score |
|---|---|---|
| **Random Forest** ✅ | **0.8324** | **0.83** |
| XGBoost | 0.8212 | 0.81 |
| PyTorch MLP | 0.8101 | 0.81 |

> **Kesimpulan Kasus 1:** Metode konvensional (Random Forest) unggul pada dataset tabular kecil (~891 baris). MLP tidak cukup data untuk belajar pola yang lebih baik.

### Catatan Penting
- Jika ada error saat feature engineering, **restart kernel dan jalankan ulang dari cell pertama** agar `train_df` dan `test_df` kembali fresh dari CSV
- Random seed `42` digunakan secara konsisten di semua model untuk reproducibility

---

## 🔢 Kasus 2 — Digit Recognizer / MNIST (Data Image)

**File:** `Kasus2.ipynb`  
**Dataset:** [Digit Recognizer (MNIST)](https://www.kaggle.com/c/digit-recognizer)  
**Task:** Multi-class classification — mengenali digit tulisan tangan 0–9

### Cara Menjalankan

1. Buka `Kasus2.ipynb` di Google Colab (**sangat direkomendasikan dengan GPU Runtime**)
   - Di Colab: `Runtime → Change runtime type → T4 GPU`
2. Jalankan cell instalasi:
   ```python
   !pip install scikit-image
   ```
3. Set Kaggle credentials
4. **Jalankan cell secara berurutan** — perhatikan bahwa ekstraksi HOG bisa memakan waktu 1–2 menit

### Alur Notebook

| Tahap | Deskripsi |
|---|---|
| **Setup & Import** | Install scikit-image, import library, download dataset MNIST dari Kaggle |
| **EDA** | Visualisasi 10 sampel gambar, distribusi kelas, rata-rata intensitas piksel per digit |
| **Preprocessing HOG+SVM** | Normalisasi, flatten ke (784,), ekstraksi HOG (orientations=9, pixels_per_cell=(7,7)) |
| **HOG + SVM** | Training SVM kernel RBF (C=10), evaluasi pada 8.400 data validasi |
| **Preprocessing CNN** | Reshape ke (-1, 28, 28, 1), normalisasi ÷ 255.0 |
| **Deep CNN** | Arsitektur multi-block: 2× Conv-BatchNorm-MaxPool-Dropout, lalu Dense-Dropout-Softmax |
| **Training CNN** | 20 epoch, Adam optimizer, EarlyStopping (patience=5), batch size 64 |
| **Evaluasi** | Confusion matrix, error analysis (contoh 5 gambar salah), perbandingan inferensi time |
| **Submission** | Generate `submission_kasus2.csv` untuk Kaggle |

### Arsitektur Deep CNN

```
Input (28×28×1)
  → Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
  → Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
  → Flatten → Dense(128) → BatchNorm → Dropout(0.5)
  → Dense(10, softmax)
```

### Hasil

| Model | Accuracy | F1-score | Salah Prediksi |
|---|---|---|---|
| **Deep CNN** ✅ | **0.9943** | **0.99** | 48 dari 8.400 |
| HOG + SVM | 0.9837 | 0.98 | 138 dari 8.400 |

> **Kesimpulan Kasus 2:** Deep learning (CNN) unggul signifikan pada data citra — mengurangi error sebesar 65.2%. CNN mampu menangkap struktur spasial hierarkis yang tidak bisa dilakukan HOG secara adaptif.

### Catatan Penting
- Ekstraksi HOG membutuhkan waktu beberapa menit — tunggu hingga selesai sebelum lanjut ke cell berikutnya
- Pastikan `deep_cnn` sudah ter-assign ke `cnn_model` sebelum menjalankan cell evaluasi perbandingan
- Training CNN pada CPU akan sangat lambat — gunakan GPU Runtime di Colab

---

## 🐦 Kasus 3 — Disaster Tweets (Data Teks)

**File:** `Kasus3.ipynb`  
**Dataset:** [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)  
**Task:** Binary classification — apakah sebuah tweet membahas bencana nyata atau tidak

### Cara Menjalankan

1. Buka `Kasus3.ipynb` di Google Colab atau Jupyter
2. Tidak perlu instalasi tambahan — semua library sudah tersedia di Colab
3. Set Kaggle credentials
4. **Jalankan semua cell berurutan** — variabel hasil training di cell konvensional dipakai di cell perbandingan akhir

### Alur Notebook

| Tahap | Deskripsi |
|---|---|
| **Setup & Import** | Set random seed (42), import sklearn/tensorflow/kagglehub, download dataset |
| **EDA** | Distribusi label target, distribusi panjang teks |
| **Text Cleaning** | Lowercase, hapus URL/mention/hashtag/karakter non-huruf, hapus spasi berlebih |
| **Train-Val Split** | 80:20 stratified split dengan `random_state=42` |
| **TF-IDF Unigram** | `TfidfVectorizer(ngram_range=(1,1), max_features=10000)` |
| **Logistic Regression** | Tuning C={0.1, 1, 10}, best C=1, F1=0.7723 |
| **TF-IDF Bigram** | `TfidfVectorizer(ngram_range=(1,2), max_features=15000)` |
| **Linear SVM** | Tuning C={0.1, 1, 10}, best C=0.1 |
| **Multinomial NB** | Tuning alpha={0.1, 0.5, 1.0}, best alpha=0.5 |
| **LSTM** | Tokenizer → Padding (max_len=50) → Embedding(10000, 128) → LSTM(64) → Dense |
| **Training LSTM** | 10 epoch, EarlyStopping (patience=3, monitor='val_loss'), berhenti di epoch 4 |
| **Evaluasi Lengkap** | DataFrame perbandingan semua model (accuracy, F1, training time) |
| **Error Analysis** | Analisis tweet salah prediksi untuk LR, SVM, dan LSTM |

### Arsitektur LSTM

```
Input (tokenized tweet, max_len=50)
  → Embedding(vocab=10000, dim=128)
  → LSTM(64 units)
  → Dropout(0.5)
  → Dense(32, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid)
```

### Hasil

| Model | Accuracy | F1-score | Training Time |
|---|---|---|---|
| **TF-IDF + Logistic Regression** ✅ | **0.8188** | **0.7723** | 0.187 s |
| TF-IDF + Multinomial NB | 0.8188 | 0.7600 | 0.005 s |
| TF-IDF + LinearSVM | 0.7978 | 0.7594 | 0.048 s |
| Embedding + LSTM | 0.5706 | 0.0000 | 21.67 s |

> **Kesimpulan Kasus 3:** LSTM gagal total — memprediksi semua sampel sebagai kelas 0 karena kurang data training dan tidak ada pretrained embedding. TF-IDF + Logistic Regression tetap menjadi baseline terkuat untuk teks tweet pendek.

### Catatan Penting
- Jika ingin meningkatkan performa LSTM: gunakan pretrained embedding (GloVe/FastText) atau fine-tune DistilBERT
- Tambahkan `class_weight='balanced'` pada model konvensional jika ingin handling imbalance lebih baik

---

## 📊 Ringkasan Perbandingan Lintas Kasus

| Kasus | Tipe Data | Model Terbaik | Metode | Accuracy | F1 |
|---|---|---|---|---|---|
| Titanic | Tabular kecil | Random Forest | Konvensional | 0.8324 | 0.83 |
| Digit Recognizer | Image | Deep CNN | Deep Learning | 0.9943 | 0.99 |
| Disaster Tweets | Teks pendek | TF-IDF + LR | Konvensional | 0.8188 | 0.7723 |

### Kapan Pakai Apa?

- **Data tabular kecil (<10k baris)** → Random Forest / XGBoost
- **Data citra** → CNN (dengan BatchNorm + Dropout)
- **Teks pendek, dataset terbatas** → TF-IDF + Logistic Regression / SVM
- **Teks, dataset besar** → Pretrained model (BERT, DistilBERT)

---

## 🔁 Reproducibility

Seluruh eksperimen menggunakan **random seed = 42** secara konsisten:
```python
# Python/NumPy/sklearn
random_state=42

# PyTorch (Kasus 1)
torch.manual_seed(42)

# TensorFlow (Kasus 2 & 3)
SEED = 42
tf.random.set_seed(SEED)
```

---

## 📚 Referensi

- Alzubaidi et al. (2021). Review of deep learning: concepts, CNN architectures. *Journal of Big Data*
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*
- Kaggle Titanic Competition — https://www.kaggle.com/c/titanic
- Kaggle Digit Recognizer — https://www.kaggle.com/c/digit-recognizer
- Kaggle NLP Disaster Tweets — https://www.kaggle.com/c/nlp-getting-started
- Stanford CS231n — https://cs231n.github.io/convolutional-networks/
