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
│   ├── autodif
│   │   ├── ffnn.py         # Implementasi utama FFNN
│   │   ├── layer.py        # Implementasi kelas Layer
│   │   ├── varValue.py     # Fungsi utilitas tambahan
│   │   └── notebook.ipynb  # Notebook untuk pengujian model
│   ├── basic
│   │   ├── ANN_basic.ipynb # Notebook untuk pengujian model
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
pip install numpy scikit-learn matplotlib
```

## Menjalankan Proyek
1. Untuk menggunakan implementasi FFNN:
```python
from src.autodiff.ffnn import FFNN

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
Buka dan jalankan `notebook.ipynb` di Jupyter Notebook

## Pembagian Tugas
| NIM | Nama | Peran |
| --- | --- | --- |
| 13522013 | Denise Felicia Tiowanni | XXXXX | 
| 13522083 | Evelyn Yosiana | XXXXX |
| 13522103 | Steven Tjhia | XXXXX |