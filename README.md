---
title: KTP Extractor API
emoji: 🪪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# ktp_extractor

Production-grade, CPU-only KTP (Indonesian ID card) field extractor built on
**PaddleOCR PP-OCRv5**. No GPU required.

## Why this exists

Most open-source KTP OCR scripts (Tesseract or PaddleOCR based) concatenate
all detected text into one flat string and then guess field boundaries by
array index (`data[nik_index + 1]`, etc.). That breaks the moment OCR
misreads a label, skips a line, or the card is slightly tilted, and there's
no way to tell a correct extraction from a wrong one after the fact.

This package instead:

1. **Reconstructs the KTP's row layout** from OCR bounding boxes (`layout.py`),
   so label and value are matched by their actual on-card position, not by
   array position in a flattened list.
2. **Matches labels with typo tolerance** (`fields.py`) using a similarity
   ratio anchored to each label's own word count -- tolerant of OCR
   misreads like `NIK` → `N1K`, without the false-positive collisions a
   looser fuzzy match produces between short labels (verified in tests).
3. **Handles multiple label:value pairs sharing one physical row**, which
   real KTPs do for `Jenis Kelamin` / `Gol. Darah`.
4. **Validates the NIK structurally** (16 digits, plausible birth date
   encoded in positions 7-12, with the +40 female-day offset), and repairs
   common OCR digit/letter confusions (`O`↔`0`, `I`/`l`↔`1`, `S`↔`5`) before
   validating.
5. **Reports confidence per field**, plus which fields are missing or
   low-confidence, so a caller can route uncertain extractions to manual
   review instead of trusting them blindly.

## 🚀 Quick Start (Panduan Cepat Setelah Clone)

File gambar sampel **`ktp1.jpg`** sudah disertakan langsung di dalam repository ini sehingga Anda bisa langsung mencoba ekstraksi tanpa perlu menyiapkan foto KTP sendiri.

---

### 1. Clone Repository & Setup Lingkungan

Pilih instruksi sesuai sistem operasi yang Anda gunakan:

#### 🍎 Pengguna macOS (Apple Silicon M1/M2/M3/M4 & Intel)

```bash
# 1. Clone repository (atau download ZIP)
git clone https://github.com/iqbalnova/ktp-exctractor-paddle-ocr.git
cd ktp-exctractor-paddle-ocr

# 2. Buat virtual environment
python3 -m venv venv

# 3. Aktifkan virtual environment
source venv/bin/activate

# 4. Install dependencies (CPU-only PaddlePaddle & FastAPI)
pip install -r requirements.txt
```

> **Catatan Apple Silicon:** Pada eksekusi pertama, PaddleOCR akan otomatis mengunduh bobot model PP-OCRv5 (~150 MB) ke `~/.paddlex`. Proses inferensi selanjutnya berjalan 100% lokal dan offline.

#### 🪟 Pengguna Windows (CMD / PowerShell)

Pastikan Python 3.10 – 3.12 sudah terinstall dan dicentang opsi *"Add python.exe to PATH"*.

* **Command Prompt (CMD):**
  ```cmd
  git clone https://github.com/iqbalnova/ktp-exctractor-paddle-ocr.git
  cd ktp-exctractor-paddle-ocr
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```

* **PowerShell:**
  ```powershell
  git clone https://github.com/iqbalnova/ktp-exctractor-paddle-ocr.git
  cd ktp-exctractor-paddle-ocr
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
  *(Jika muncul script execution policy error, jalankan: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

#### 🐧 Pengguna Linux (Ubuntu / Debian)

```bash
# 1. Install library sistem yang dibutuhkan OpenCV
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0

# 2. Clone dan masuk ke folder
git clone https://github.com/iqbalnova/ktp-exctractor-paddle-ocr.git
cd ktp-exctractor-paddle-ocr

# 3. Buat dan aktifkan venv, lalu install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Cara Mencoba Ekstraksi dengan `ktp1.jpg`

Pastikan virtual environment Anda sudah aktif (`venv`), lalu pilih cara yang Anda inginkan:

#### 🔹 Opsi 1: Menjalankan Script Contoh (`example_usage.py`)
Script ini menampilkan hasil ekstraksi field utama secara langsung di terminal:
```bash
python3 example_usage.py ktp1.jpg
```
*(Di Windows gunakan: `python example_usage.py ktp1.jpg`)*

#### 🔹 Opsi 2: Menggunakan CLI (Output JSON Rapi)
Mendapatkan output JSON terstruktur lengkap dengan confidence score dan status validasi NIK:
```bash
# Tampilkan output JSON rapi di terminal:
python3 -m ktp_extractor.cli ktp1.jpg --pretty

# Atau simpan output JSON ke dalam file:
python3 -m ktp_extractor.cli ktp1.jpg > hasil_ktp1.json
```

#### 🔹 Opsi 3: Menjalankan REST API (FastAPI)
Jalankan server API lokal:
```bash
python3 app.py
# atau: uvicorn app:app --host 127.0.0.1 --port 8000
```
Setelah server berjalan:
1. Buka browser ke **Swagger UI interaktif**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
2. Klik endpoint **POST `/api/v1/extract`** $\rightarrow$ klik **Try it out** $\rightarrow$ pilih file `ktp1.jpg` $\rightarrow$ klik **Execute**.
3. Atau uji coba langsung lewat `curl` di terminal lain:
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/v1/extract" \
        -F "file=@ktp1.jpg"
   ```

*(Catatan: Jika Anda menyetel environment variable `KTP_API_KEY`, tambahkan header `-H "X-API-Key: YOUR_KEY"` pada setiap request).*

---

### 3. Menggunakan di Kode Python Anda Sendiri

```python
from ktp_extractor import KTPExtractor

# 1. Inisialisasi extractor (CPU-only)
extractor = KTPExtractor(lang="id")

# 2. Ekstrak data dari file gambar
record = extractor.extract("ktp1.jpg")

# 3. Akses nilai field secara langsung
print("Nama            :", record.nama.value)
print("NIK             :", record.nik.value, f"(Valid: {record.nik_valid})")
print("Tempat/Tgl Lahir:", record.tempat_tgl_lahir.value)
print("Alamat          :", record.alamat.value)
print("RT / RW         :", record.rt_rw.value)
print("Kab/Kota        :", record.kabupaten_kota.value)
print("Provinsi        :", record.provinsi.value)

# 4. Export seluruh hasil ke format JSON
print(record.to_json(indent=2))

# 5. Cek field yang membutuhkan tinjauan manual jika confidence rendah
if record.low_confidence_fields:
    print("Field perlu dicek manual:", record.low_confidence_fields)
```

## Design notes / known limitations

- **This is a rules-based extractor on top of a generic OCR engine**, not a
  document-understanding model. It knows the *shape* of a standard KTP
  (label, then value, on the same row) but has no learned model of the
  layout the way a fine-tuned Donut/LayoutLM model would. For very degraded
  photos this is more predictable to debug, but a fine-tuned VLM (see
  `emisilab/model-ocr-ktp-v1` or `rizkynindra/donut-ktp` on Hugging Face)
  may score higher on badly-lit or heavily-tilted photos, at the cost of
  being much slower on CPU.
- **Fuzzy label matching intentionally will not catch every possible typo**
  on short labels (`Agama`, `Nama`, `Alamat`, `NIK`): the similarity
  threshold is set high enough to avoid those short labels being confused
  with each other, which means some single-character misreads on those
  specific fields won't be caught by fuzzy matching alone. Add observed
  misreadings to that label's `aliases` tuple in `fields.py` as you
  encounter them in production -- this is the intended extension point,
  rather than patching the parsing logic itself.
- **No image preprocessing (deskew/contrast) is bundled.** The row-grouping
  in `layout.py` tolerates a few degrees of tilt via vertical-overlap
  matching, but a badly lit or heavily rotated photo should be corrected
  (or re-captured) before calling `extract()`.
- **NIK validation is structural, not authoritative.** There is no public
  official checksum digit; this checks digit count, month range, and a
  plausible day-of-month, which catches most OCR garbling but cannot
  confirm the NIK belongs to a real, registered person.
- **PaddleOCR 3.x CPU memory usage**: some 3.x releases have had reported
  CPU memory regressions on certain inputs/configs (see
  `PaddlePaddle/PaddleOCR#17955`). If you see unexpectedly high memory use
  in production, try `enable_mkldnn=False` and/or lower `cpu_threads`, and
  pin a known-good `paddleocr`/`paddlepaddle` version pair as in
  `requirements.txt`.

## Project layout

```
ktp_extractor/
├── models.py       # TextBox, FieldResult, KTPRecord dataclasses (no deps)
├── layout.py       # bounding-box -> reading-order rows (no deps)
├── validators.py   # NIK validation, digit/RT-RW cleanup (no deps)
├── fields.py       # label catalogue + fuzzy label:value matching (no deps)
├── ocr_engine.py   # PaddleOCR PP-OCRv5 wrapper (only module needing paddleocr)
├── extractor.py    # KTPExtractor: ties the above into one pipeline
└── cli.py          # command-line entry point
tests/
└── test_extraction_logic.py   # runs against synthetic OCR output, no
                                # PaddleOCR installation required
```

Everything except `ocr_engine.py` has zero external dependencies, so the
parsing/validation logic can be (and is) unit-tested with synthetic OCR
output -- see `tests/test_extraction_logic.py`, which passes without
PaddleOCR installed.

## Running tests

Pastikan virtual environment sedang aktif:

- **macOS / Linux:**
  ```bash
  python3 tests/test_extraction_logic.py
  ```
- **Windows:**
  ```cmd
  python tests\test_extraction_logic.py
  ```

## Storage & Cleanup (Panduan Bersih-bersih)

Jika suatu saat ingin menghapus project ini dan mengembalikan kapasitas penyimpanan secara tuntas:

### 1. Rincian Penyimpanan Terpakai
- **Di Dalam Project (`venv/`)**: `~919 MB`  
  Semua package Python (`paddlepaddle`, `paddleocr`, `opencv`, dll). **Otomatis terhapus** bila folder project dihapus.
- **Di Luar Project (Folder Pengguna `~`)**:
  - `~/.paddlex`: `~196 MB` (file bobot model AI PaddleOCR).
  - Cache installer pip (`~/Library/Caches/pip` di macOS, `%LocalAppData%\pip\cache` di Windows, atau `~/.cache/pip` di Linux).

### 2. Cara Menghapus Bersih

#### 🍎 macOS & 🐧 Linux

Jalankan perintah ini di Terminal:

```bash
# 1. Bersihkan cache installer pip
pip cache purge

# 2. Hapus bobot model AI
rm -rf ~/.paddlex

# 3. Hapus virtual environment (atau hapus seluruh folder project ini)
rm -rf venv/
```

#### 🪟 Windows

Jalankan di Command Prompt (CMD):

```cmd
:: 1. Bersihkan cache installer pip
pip cache purge

:: 2. Hapus bobot model AI
rmdir /s /q %USERPROFILE%\.paddlex

:: 3. Hapus virtual environment
rmdir /s /q venv
```

Atau di PowerShell:

```powershell
# 1. Bersihkan cache installer pip
pip cache purge

# 2. Hapus bobot model AI
Remove-Item -Recurse -Force ~\.paddlex

# 3. Hapus virtual environment
Remove-Item -Recurse -Force .\venv
```
