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

## Installation (macOS / Linux)

### 1. Setup Virtual Environment (Disarankan untuk macOS)

Di macOS (khususnya pengguna Homebrew dengan proteksi PEP 668), gunakan virtual environment agar tidak mengotori sistem OS:

```bash
# 1. Buat virtual environment lokal di folder project
python3 -m venv venv

# 2. Aktifkan virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

This installs CPU-only `paddlepaddle` -- do not install `paddlepaddle-gpu`,
it isn't needed and pulls in a much larger, CUDA-dependent package.

> **Catatan untuk Apple Silicon (MacBook M1/M2/M3):**  
> `ktp_extractor` sudah diatur default untuk arsitektur CPU ARM64 (`enable_mkldnn=False`, `cpu_threads=2`, dan auto-resizing resolusi gambar).
> 
> Pemanggilan pertama kali membutuhkan koneksi internet untuk mengunduh bobot model PP-OCRv5 (~150 MB) ke `~/.paddlex`. Setelah terunduh, library berjalan sepenuhnya offline.

## Usage (Cara Menjalankan)

Pastikan virtual environment dalam keadaan aktif (`source venv/bin/activate`):

### 1. Jalankan Script Contoh Langsung (`example_usage.py`)
Cara paling cepat untuk mencoba ekstraksi:

```bash
# Menjalankan pada contoh foto KTP yang sudah ada di folder
python example_usage.py ktp1.jpeg
```

### 2. Jalankan Lewat CLI (Format JSON)
Untuk mendapatkan output terstruktur JSON lengkap:

```bash
# Menampilkan JSON rapi (--pretty)
python -m ktp_extractor.cli ktp1.jpeg --pretty

# Format satu baris (cocok untuk piping ke jq atau file log)
python -m ktp_extractor.cli ktp1.jpeg > output.json
```

### 3. Digunakan dalam Kode Python Sendiri

```python
from ktp_extractor import KTPExtractor

# Inisialisasi extractor (default CPU)
extractor = KTPExtractor(lang="id")
record = extractor.extract("ktp1.jpeg")

print("Nama :", record.nama.value)
print("NIK  :", record.nik.value, f"(Valid: {record.nik_valid})")
print("TTL  :", record.tempat_tgl_lahir.value)
print("Kota :", record.kabupaten_kota.value)

# Dump full data ke JSON
print(record.to_json(indent=2))

if record.low_confidence_fields:
    print("Perlu review:", record.low_confidence_fields)
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

```bash
python tests/test_extraction_logic.py
```

## Storage & Cleanup (Panduan Bersih-bersih)

Jika suatu saat ingin menghapus project ini dan mengembalikan kapasitas penyimpanan laptop secara tuntas:

### 1. Rincian Penyimpanan Terpakai
- **Di Dalam Project (`ktp_extractor/venv`)**: `~919 MB`  
  Semua package Python (`paddlepaddle`, `paddleocr`, `opencv`, dll). **Otomatis terhapus** bila folder project dihapus.
- **Di Luar Project (Folder User `~`)**:
  - `~/.paddlex`: `~196 MB` (file bobot model AI PaddleOCR).
  - `~/Library/Caches/pip`: `~219 MB` (cache installer sementara saat `pip install`).

### 2. Cara Menghapus Bersih (100% Tuntas Bebas Sisa)

Jalankan perintah ini di Terminal:

```bash
# 1. Bersihkan cache installer pip
pip cache purge

# 2. Hapus bobot model AI dari laptop
rm -rf ~/.paddlex

# 3. Hapus virtual environment (atau hapus seluruh folder project ini)
rm -rf venv/
```
