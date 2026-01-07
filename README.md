# SIPANDA-OCR
Sistem Pemisahan dan Penamaan Dokumen berbasis OCR untuk Dokumen Akhir Perpajakan

SIPANDA OCR adalah aplikasi berbasis Python dan Streamlit yang digunakan untuk melakukan:
- Pemisahan dokumen PDF hasil scan
- Pembacaan teks menggunakan OCR
- Penamaan file otomatis berdasarkan nomor surat
- Pengemasan hasil akhir dalam bentuk ZIP  
yang dioptimalkan untuk dokumen perpajakan seperti **STP, SKPKB, SKPN, dan SKPLB**.

---

## 📌 Fitur Utama
- Upload multiple PDF sekaligus
- Pemisahan dokumen otomatis sesuai jenis surat
- OCR halaman pertama untuk penamaan file
- Standarisasi format nama file: `NamaWP_nomor.pdf`
- Output hasil dalam bentuk ZIP
- Siap diimplementasikan ulang (*reproducible environment*)

---

## 🖥️ Spesifikasi Lingkungan Sistem

### Sistem Operasi
- Windows 10 / 11  
- Linux (Ubuntu 20.04 ke atas – opsional)

### Bahasa & Runtime
- Python **3.10** (direkomendasikan)

---

## 📦 Dependencies

Daftar library utama yang digunakan:

- streamlit
- pytesseract
- pdf2image
- pillow
- PyPDF2
- pandas
- numpy

Semua dependency dikelola melalui file `requirements.txt`.

---

## 🧰 Prasyarat Tambahan

### 1. **Tesseract OCR**
Pastikan Tesseract OCR sudah terinstal di sistem.

**Windows:**
- Download dari: https://github.com/UB-Mannheim/tesseract/wiki
- Tambahkan path instalasi ke Environment Variable

Contoh path:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```
### 2. Poppler (Untuk pdf2image)
Windows:
- Download Poppler for Windows
- Tambahkan folder bin ke PATH

Contoh path:
```
C:\Users\HP\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin
```

----

## ⚙️ Instalasi Aplikasi

### 1. Clone Repository
```
git clone https://github.com/ranydwi/SIPANDA-OCR.git
cd SIPANDA-OCR
```

### 2. Buat Virtual Environment (Direkomendasikan)
Pada command prompt / terminal :
```
conda create -n sipandaocr python=3.10
```

### 3. Aktifkan Environment
Pada command prompt / terminal :
```
conda activate sipanda-ocr
```
Jika berhasil, prompt akan berubah menjadi `sipandaocr`

### 4. Install Dependencies

Masuk ke direktori project, lalu jalankan:
```
pip install -r requirements.txt
```
📌 Disarankan tetap menggunakan pip untuk menjaga konsistensi dependency Python, meskipun environment dibuat dengan Conda.

---
## 🧾 Konfigurasi Sistem
Tambahkan file `config.toml` pada folder tempat Streamlit berasal.

```
~/.streamlit/config.toml
```
Biasanya ada di directory Program Files di Device.

---
## ▶️ Jalankan Aplikasi
```
streamlit run app.py
```

Aplikasi akan berjalan di:

http://localhost:8080

