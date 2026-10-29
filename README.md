# Deteksi Ekspresi Wajah (FER-2013) — Proyek UTS Deep Learning

Proyek kelompok mata kuliah Deep Learning, tahap UTS (fondasi dan baseline): klasifikasi 7 ekspresi wajah
(angry, disgust, fear, happy, neutral, sad, surprise) pada citra grayscale 48x48 menggunakan
**MLP** dan **CNN sederhana** (PyTorch).

## Anggota dan peran
1	Tulus Andriansah	2330105030002
2	Meisya Pradana Mentari	2330105030020
3	Alvin Nugraha Tangkawata	2330305030085
4	Arbiyannoor Ilham Sanusie	223020503097
5	Christian Iuliano	203010503003

## Dataset
- Nama: FER-2013 (versi folder gambar di Kaggle), https://www.kaggle.com/datasets/msambare/fer2013
- Isi: 28.709 citra train dan 7.178 citra test, 7 kelas, semua 48x48 grayscale (JPEG).
- Tidak ada validation bawaan: validation diambil 10% dari train secara stratified (seed 42) sebelum pelatihan.
- **Temuan kualitas data:** 531 hash isi piksel muncul di train dan test sekaligus (703 citra train identik dengan citra test),
  ditambah duplikat internal dan 39 hash berlabel bertentangan. Citra train yang identik dengan test, duplikat, dan konflik label
  dikeluarkan dari data latih (`HAPUS_BOCOR = True`); test tidak diubah. Hasil split: train 24.220, validation 2.692, test 7.178.

## Struktur repositori
```
notebooks/   notebook utama (Data Engineer + MLP + CNN + evaluasi)
models/      bobot model terlatih (.pt)
reports/     figures/ (grafik) dan logs/ (log pelatihan, tabel hasil, classification report)
data/        catatan dataset (dataset tidak di-commit)
src/         disiapkan untuk modul .py pada tahap UAS
```

## Cara menjalankan (Google Colab)
1. Unduh dataset dari tautan Kaggle di atas (file ZIP).
2. Buka `notebooks/UTS_DL_Deteksi_Ekspresi_Wajah_Final.ipynb` di Google Colab.
3. Pilih **Runtime > Change runtime type > T4 GPU**.
4. Jalankan seluruh sel dari atas (**Runtime > Run all**). Pada Tahap 1, unggah file ZIP dataset saat diminta.
5. Setel `QUICK_TEST = True` di Tahap 13a untuk uji cepat (2 epoch); gunakan `False` untuk hasil penuh.
6. Hasil (bobot, log, grafik) dibuat di `/content/hasil_uts` dan diunduh sebagai ZIP pada Tahap 27.

Dependensi: lihat `requirements.txt` (Colab sudah menyediakan sebagian besar library).

## Hasil ringkas (test set, satu run, seed 42)
| Model | Parameter | Akurasi | Macro-F1 | Val macro-F1 |
|---|---|---|---|---|
| MLP | 1.313.287 | 0,4207 | 0,3740 | 0,3693 |
| CNN sederhana | 1.468.135 | 0,5529 | 0,5211 | 0,5088 |

Model dipilih berdasarkan macro-F1 validation; test dipakai satu kali untuk evaluasi akhir.
Angka dapat berbeda tipis jika notebook dijalankan ulang; rujukan resmi: `reports/logs/hasil_test.csv`.

## Keterbatasan
- Dataset tidak seimbang (kelas happy sekitar 20 kali lebih banyak daripada disgust pada data latih); kelas fear dan sad sulit dibedakan.
- CNN sederhana overfitting (akurasi train jauh di atas validation); belum ada regularisasi pada baseline.
- Satu run per model dengan seed tetap; selisih kecil antar model belum tentu bermakna.
- Hyperparameter belum dituning; label FER-2013 berisik dan citra hanya 48x48.

## Rencana lanjutan (UAS)
Transfer learning dan fine-tuning, regularisasi dan penanganan ketidakseimbangan kelas, hyperparameter tuning, studi ablasi,
interpretabilitas (Grad-CAM), analisis bias, dan demo aplikasi.
