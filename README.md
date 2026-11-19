# HPAI - Dashboard Analisis Avian Influenza

Project ini sudah disiapkan untuk deploy ke **Streamlit Community Cloud**.
Dataset `Avian Influenza (HPAI).csv` disertakan sebagai dataset bawaan, sehingga
aplikasi langsung memiliki data saat dibuka. Pengguna tetap dapat mengunggah CSV
lain dari sidebar untuk mengganti dataset bawaan.

## Deploy online

1. Ekstrak ZIP ini.
2. Buat repository baru di GitHub.
3. Upload seluruh isi folder ini ke root repository (jangan upload folder ZIP-nya).
4. Buka https://share.streamlit.io/ dan login menggunakan GitHub.
5. Pilih **Create app**.
6. Pilih repository, branch `main`, dan file utama `app.py`.
7. Klik **Deploy**. Tunggu proses build selesai; Streamlit akan memberi URL publik.

## Menjalankan di komputer

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Dataset bawaan

`Avian Influenza (HPAI).csv` digunakan secara otomatis saat aplikasi dibuka.
Unggah CSV lain dari sidebar bila ingin mengganti dataset.

Kolom wajib:
`Year`, `Month`, `Day`, `Time`, `Latitude`, `Longitude`, `State`, `County`,
`target_H5_HPAI`.

## Catatan privasi

Dataset yang disertakan di repository dapat diakses oleh siapa pun yang memiliki
akses ke repository atau aplikasi sesuai pengaturan publikasinya. Jangan commit
data pribadi, rahasia, atau data yang tidak memiliki izin untuk dipublikasikan.
Untuk dataset sensitif, hapus CSV dari repository dan gunakan upload atau sumber
data privat yang dikonfigurasi secara aman.

## Catatan analisis

Model memakai RandomForestClassifier dan membagi data secara stratified.
Metrik yang ditampilkan adalah hasil pada test split, bukan jaminan performa
pada data baru. Aplikasi bukan alat diagnosis atau sistem peringatan resmi.
