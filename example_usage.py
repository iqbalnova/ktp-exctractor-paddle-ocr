"""
Minimal example: extract fields from a KTP image and inspect the result.

Run:
    python example_usage.py path/to/ktp_photo.jpg
"""
import sys

from ktp_extractor import KTPExtractor, KTPExtractionError


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-ktp-image>")
        return 1

    image_path = sys.argv[1]

    # First run downloads the PP-OCRv5 mobile models (a few tens of MB) and
    # caches them under ~/.paddlex; subsequent runs are instant to construct.
    extractor = KTPExtractor(lang="id", cpu_threads=4)

    try:
        record = extractor.extract(image_path)
    except KTPExtractionError as e:
        print(f"Could not read this image at all: {e}")
        return 1

    print(f"Nama            : {record.nama.value}")
    print(f"NIK             : {record.nik.value}  (valid: {record.nik_valid})")
    print(f"Tempat/Tgl Lahir: {record.tempat_tgl_lahir.value}")
    print(f"Alamat          : {record.alamat.value}, RT/RW {record.rt_rw.value}")
    print(f"Kel/Desa        : {record.kel_desa.value}, Kec. {record.kecamatan.value}")
    print(f"Kab/Kota        : {record.kabupaten_kota.value}, {record.provinsi.value}")
    print(f"\nOverall confidence: {record.overall_confidence:.2f}")

    if record.missing_fields:
        print(f"Fields not detected: {record.missing_fields}")
    if record.low_confidence_fields:
        print(f"Fields flagged for manual review: {record.low_confidence_fields}")
    if not record.nik_valid:
        print(f"NIK validation issue: {record.nik_validation_notes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
