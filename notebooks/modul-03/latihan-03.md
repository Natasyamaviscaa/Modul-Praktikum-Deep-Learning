# Latihan Mandiri di Lab — Modul 03

**Nama:** Natasya Amavisca
**NIM:** 123450024
**Kelas:** Deep Learning RB
**Tanggal:** 2026-09-30
**Notebook:** `M03_123450024.ipynb` (sel Latihan, `df_bagian_f`)
**Seed:** 24 (4 digit terakhir NIM) · **Device:** CPU · **Protokol:** 12.000 latih / 3.000 validasi, FNN 784 → 128 → 10, batch 128, 5 epoch

## 1. Pertanyaan

Jelaskan mengapa batch 32 menghasilkan 1.875 update sedangkan batch 512 hanya 120 update untuk anggaran epoch yang sama. Tuliskan lebih dahulu prediksi tentang mana yang mencapai validation loss lebih rendah dan mana yang lebih cepat per epoch, lalu bandingkan dengan angka yang diperoleh.

## 2. Prediksi (ditulis sebelum melihat angka)

| No | Prediksi | Alasan |
|---|---|---|
| 1 | Batch 32 = 1.875 update, batch 512 = 120 update | Satu epoch dibagi menjadi $\lceil n/\text{batch}\rceil$ batch, lalu dikalikan 5 epoch. |
| 2 | Batch 32 mencapai validation loss **lebih rendah** (sekitar 0,38–0,40) | Pada anggaran epoch yang sama ia memberi 1.875 langkah optimizer berbanding hanya 120 langkah, sehingga batch 512 diperkirakan belum konvergen. |
| 3 | Batch 512 **lebih cepat per epoch** (sekitar 2–3 kali) | Hanya 24 iterasi per epoch berbanding 375; tiap iterasi lebih berat, tetapi overhead pemanggilan loop jauh lebih sedikit. |

## 3. Hasil pengukuran

| Butir | Prediksi | Hasil aktual | Kesesuaian |
|---|---|---|---|
| Update per epoch batch 32 | 375 | 375 = $\lceil12000/32\rceil$ | Sesuai |
| Update total batch 32 | 1.875 | 1.875 (375 × 5), `cocok=True` | Sesuai |
| Update per epoch batch 512 | 24 | 24 = $\lceil12000/512\rceil$ | Sesuai |
| Update total batch 512 | 120 | 120 (24 × 5), `cocok=True` | Sesuai |
| Validation loss | batch 32 lebih rendah | **0,3873** (b32) vs **0,4477** (b512) | Sesuai |
| Validation accuracy | mengikuti loss | 0,8633 (b32) vs 0,8400 (b512) | Sesuai |
| Runtime per epoch | batch 512 lebih cepat | 0,6079 s (b32) vs 0,2117 s (b512) | Sesuai |
| Selisih kecepatan | 2–3 kali | 2,9 kali (0,6079 / 0,2117) | Sesuai |
| Runtime total 5 epoch | batch 512 lebih hemat | 3,04 s (b32) vs 1,06 s (b512) | Sesuai |

## 4. Perbandingan dan pembahasan

**Mengapa 1.875 vs 120 update.** Anggaran epoch menentukan berapa kali *seluruh* data latih dilewati, bukan berapa kali optimizer dipanggil. Satu epoch dipotong menjadi $\lceil12000/32\rceil = 375$ batch dan $\lceil12000/512\rceil = 24$ batch; dikalikan 5 epoch menjadi 1.875 dan 120 update. Karena kedua run memproses 12.000 contoh yang sama per epoch, selisihnya murni jumlah langkah parameter: batch 32 melakukan 15,6 kali lebih banyak update.

**Validation loss.** Prediksi terbukti: batch 32 unggul 0,0604 (0,3873 vs 0,4477) dan accuracy-nya unggul 0,0233 (0,8633 vs 0,8400). Hal ini konsisten dengan anggaran update yang 15,6 kali lebih besar pada epoch yang sama.

**Kecepatan.** Prediksi juga terbukti: batch 512 menyelesaikan satu epoch dalam 0,2117 detik berbanding 0,6079 detik untuk batch 32 (2,9 kali lebih cepat), karena satu epoch hanya membutuhkan 24 iterasi berbanding 375.

**Trade-off.** Batch kecil menang dalam satuan *epoch*, batch besar menang dalam satuan *waktu*. Pada anggaran waktu 3,04 detik (yang dipakai batch 32 untuk 5 epoch), batch 512 mampu menjalankan ±14 epoch dan batch 128 ±9 epoch. Karena itu, bila batas komputasi diukur dalam detik, batch besar sebaiknya diberi lebih banyak epoch, sedangkan bila batasnya diukur dalam epoch, batch kecil memberi hasil sedikit lebih baik dengan biaya waktu yang jauh lebih tinggi.
