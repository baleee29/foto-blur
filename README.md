# FotoBlur

FotoBlur adalah efek webcam real-time seperti tren TikTok: kamera tampil normal, lalu perlahan menjadi blur ketika pose tangan angka 2 / peace sign terdeteksi. Saat pose hilang, blur akan kembali normal secara perlahan.

Jika love sign terdeteksi, foto `Lovesign.jpeg` akan muncul sebagai overlay di tampilan kamera.

Project ini punya dua versi:

- Website statis untuk deploy ke Vercel.
- Aplikasi Python lokal dengan OpenCV.

## Website

File utama website:

- `index.html`
- `styles.css`
- `script.js`
- `assets/Lovesign.jpeg`
- `hand_landmarker.task`
- `vercel.json`
- `.vercelignore`

Jalankan lokal dengan server statis:

```bash
python -m http.server 5173
```

Buka:

```text
http://127.0.0.1:5173
```

Browser membutuhkan HTTPS atau localhost agar akses kamera bisa berjalan. Saat sudah di Vercel, URL deploy memakai HTTPS.

Di handphone, efek blur memakai fallback canvas yang lebih kompatibel dengan browser mobile. Deteksi tangan tetap memakai MediaPipe Tasks.

## Deploy ke Vercel

1. Push project ke repository GitHub bernama `foto-blur`.
2. Buka Vercel dan pilih `Add New Project`.
3. Import repository `foto-blur`.
4. Gunakan pengaturan default untuk static site.
5. Deploy.

Catatan: file Python desktop disimpan di folder `desktop-python/` dan diabaikan oleh Vercel melalui `.vercelignore`, jadi Vercel hanya membaca website statis di root project.

## GitHub

Contoh command untuk membuat repository lokal:

```bash
git init
git add .
git commit -m "Initial FotoBlur web app"
git branch -M main
git remote add origin https://github.com/USERNAME/foto-blur.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub kamu.

## Versi Python

Teknologi:

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy

Install dependency Python:

```bash
pip install -r desktop-python/requirements.txt
```

Jalankan aplikasi Python:

```bash
python desktop-python/main.py
```

Saat pertama kali dijalankan, aplikasi akan memakai file model `hand_landmarker.task` di folder proyek. Jika file belum ada, aplikasi akan mencoba mengunduhnya otomatis.

## Cara Menggunakan

Buka aplikasi, hadapkan tangan ke kamera, lalu buat pose angka 2 / peace sign. Kamera akan blur secara perlahan setelah pose terdeteksi beberapa frame berturut-turut.

Saat pose tangan hilang, efek blur akan turun perlahan sampai tampilan kamera kembali normal.

## Tombol Keluar

Tekan `Q` untuk keluar dari aplikasi.
