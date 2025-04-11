# Feedforward Neural Network (FFNN) Implementation

## Deskripsi Proyek
Repository ini berisi implementasi FFNN from scratch. Proyek ini berfokus pada implementasi FFNN yang fleksibel dengan berbagai fungsi aktivasi, fungsi loss, metode inisialisasi bobot, dan kemampuan pelatihan.

## Fitur
- Implementasi FFNN from scratch
- Mendukung berbagai fungsi aktivasi:
  - Linear
  - ReLU
  - Sigmoid
  - Hiperbolik Tangen (tanh)
  - Softmax
- Fungsi loss:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
- Metode inisialisasi bobot:
  - Inisialisasi nol
  - Distribusi random uniform
  - Distribusi random normal
- Pelatihan berbasis gradient descent
- Kemampuan visualisasi dan analisis model

## Project
```
repository/
│
├── src/
│   ├── autodiff
│   │   ├── ffnn.py         # Implementasi utama FFNN
│   │   ├── layer.py        # Implementasi kelas Layer
│   │   ├── varValue.py     # Fungsi utilitas tambahan
│   │   └── notebook.ipynb  # Notebook untuk pengujian model dengan autodiff
│   ├── basic
│   │   ├── FFNN_basic_class.py         # Implementasi utama FFNN basic
│   │   ├── FFNN_basic.ipynb            # Notebook untuk pengujian model basic
├── doc/
│   └── Tubes1_13522013_13522083_13522103.pdf     # Laporan
│
└── README.md
```

## Setup dan Instalasi
### Prasyarat
- Python
- Library:
  - NumPy
  - scikit-learn
  - Matplotlib (untuk visualisasi)
  - Jupyter Notebook (opsional, untuk menjalankan notebook.ipynb)
  - tqdm (untuk progress bar)
  - pickle (untuk menyimpan model)
  - networkx (untuk visualisasi arsitektur jaringan)
  - seaborn (untuk visualisasi data)

### Langkah Instalasi
1. Clone repository
```bash
git clone https://github.com/evelynnn04/ML-Tubes1-K33.git
cd ML-Tubes1-K33
```

2. Buat environment virtual (direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate  # Pada Windows, gunakan `venv\Scripts\activate`
```

3. Install package yang diperlukan
```bash
pip install numpy scikit-learn matplotlib tqdm networkx seaborn
```

## Menjalankan Proyek

### Menjalankan FFNN dengan AutoDiff
1. Untuk menggunakan implementasi FFNN dengan AutoDiff:
```python
from src.autodif.ffnn import FFNN
from src.autodif.layer import Layer

# Contoh
model_ffnn = FFNN(
    loss='cce',
    batch_size=1,
    learning_rate=0.1,
    epochs=2,
    verbose=1,
)
model_ffnn.build_layers(
    Layer(n_neurons=4, init='uniform', activation='relu'),
    Layer(n_neurons=10, init='uniform', activation='softmax')
)
model_ffnn.fit(X_train, y_train)
```

2. Untuk menjalankan eksperimen:
Buka dan jalankan `src/autodiff/notebook.ipynb` di Jupyter Notebook

### Menjalankan FFNN Basic
1. Untuk menggunakan implementasi FFNN basic:
```python
from src.basic.FFNN_basic_class import FFNN

# Contoh
model_ffnn = FFNN(
    loss='cce',
    batch_size=200,
    learning_rate=0.01,
    epochs=5,
    verbose=1,
    l1_lambda=0.1, 
    l2_lambda=0.1
)
model_ffnn.fit(X_train, y_train)
predictions = model_ffnn.predict(X_test)
```

2. Untuk menjalankan eksperimen dengan FFNN basic:
Buka dan jalankan `src/basic/FFNN_basic.ipynb` di Jupyter Notebook

## Pembagian Tugas
| Kegiatan | Nama (NIM) |
| --- | --- |
| Desain struktur kelas (VarValue, Layer, FFNN) | Steven Tjhia (13522103) |
| Implementasi fungsi forward() di Layer | Steven Tjhia (13522103) |
| Implementasi fungsi backward() dan update bobot | Denise Felicia Tiowanni (13522013) |
| Penanganan fungsi aktivasi (ReLU, Sigmoid, Softmax, dll) | Denise Felicia Tiowanni (13522013), Evelyn Yosiana (13522083), Steven Tjhia (13522103) |
| Pembuatan fungsi loss (MSE, BCE, CCE) | Denise Felicia Tiowanni (13522013), Evelyn Yosiana (13522083), Steven Tjhia (13522103) |
| Implementasi proses training (fit()) dan prediksi (predict()) | Evelyn Yosiana (13522083) |
| Visualisasi arsitektur jaringan (visualize()) | Evelyn Yosiana (13522083) |
| Visualisasi distribusi bobot dan gradien dan grafik loss per epoch | Denise Felicia Tiowanni (13522013) |
| Regularisasi | Evelyn Yosiana (13522083) |
| Pembuatan Laporan | Denise Felicia Tiowanni (13522013), Evelyn Yosiana (13522083), Steven Tjhia (13522103) |