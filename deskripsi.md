```
Tugas Pemrograman 2, Kuliah TBI
Search Engine “from Scratch”
```
Pada Tugas Pemrograman 2 kali ini, Anda ditugaskan untuk membuat _search engine_
dari awal, dengan hanya menggunakan _library_ standar pada python, **bukan** _library
search engine_ siap pakai. Tujuannya adalah agar Anda bisa memahami konsep dan
teori yang sudah Anda pelajari sejauh ini ke dalam bentuk implementasi.

**Anda tidak mulai dari nol (jadi “nggak** **_from scratch_** **banget lah ya”)!** Anda sudah
diberikan kode untuk memulai eksplorasi terkait hal ini. Tugas Anda adalah
menggunakan **kreatifitas** Anda untuk memodifikasi dan/atau menambahkan fitur-
fitur dari kode yang sudah ada.

**Deliverables:**

- Di TP2 kali ini, Anda wajib menyimpan kode Anda di **github** Anda masing-
    masing;
- Anda wajib menuliskan dokumentasi program Anda dalam **Bahasa Inggris**
    dan menuliskan highlight fitur-fitur penting dan cara _run_ /penggunaan dari
    program Anda;
- Tidak ada yang perlu disubmit ke SCELE. Pengumpulannya adalah dengan
    kirim email Pak Alfan dan menyampaikan **link ke github** untuk tugas TP2 ini.
- **Deadline: Senin 30 Maret 2026, Pukul 18:00 sore**

* Mengapa kami menyuruh Anda untuk mengerjakan Tugas Pemrograman 2 ini
menggunakan github? Kami ingin agar Anda mempunyai portfolio yang baik kektika
nanti setelah lulus. Sebagai seorang lulusan Fasilkom UI, isi github Anda sangat
berharga dan menentukan karir Anda di masa mendatang. Isilah github kalian
dengan proyek-proyek _open source_ yang berkualitas dan membanggakan.


**Penjelasan Kode Awal yang Diberikan**

Secara umum, Anda diberikan:

- 3 buah folder: collection, index, tmp
- 2 buah file *.txt: qrels.txt dan queries.txt
- 6 buah file *.py: bsbi.py, compression.py, evaluation.py, index.py, search.py,
    util.py

Folder **collection** berisi koleksi dokumen yang akan di-indeks. Sebuah dokumen
disimpan sebagai sebuah file teks di folder ini. Nantinya, Anda boleh mengganti isi
koleksi yang ada saat ini dengan koleksi lain yang Anda sukai.

Folder **index** awalnya kosong. Nantinya, folder ini akan berisi file biner yang berisi
**“inverted index”** dari koleksi dokumen yang ada pada folder **collection**.

Folder **tmp** digunakan saat proses _indexing_ untuk menyimpan **_“temporary inverted
index”_** sebelum akhirnya digabung menjadi satu **_inverted index_** pada folder **index**.

File **qrels.txt** dan **queries.txt** digunakan untuk melakukan evaluasi kualitas efektivitas
dari _search engine_ yang sudah dikembangkan. File **qrels.txt** berisi daftar ID _query_ dan
juga ID dokumen yang diketahui **relevan**. Sekedar informasi, informasi relevansi ini
diberikan oleh annotator manuasia. Kemudian, file **queries.txt** berisi daftar 30 _query_
yang digunakan untuk menguji _search engine_ Anda.

Sisanya adalah file *.py yang menjadi jantung dari _search engine_ kita. **Setiap file sudah
diberikan dokumentasi yang jelas untuk setiap method/fungsi yang ada**. Berikut
adalah diagram kebergantungan antar modul python tersebut.

```
index.py ---------------
\
--- bsbi.py-----
/ / \
util.py-------------- / -----------------
/ | evaluation.py |
/ | search.py |
/ -----------------
/ /
compression.py ------------------------
```

File **bsbi.py** berisi abstraksi **Inverted Index** dan juga logika untuk melakukan
indexing dengan skema **BSBI** ( **blocked-sort based indexing** ). File **index.py** berisi
logika dasar untuk menulis dan membaca indeks pada **bsbi.py**.

File **util.py** berisi utilitas dasar untuk melakukan indexing, seperti class **IdMap** untuk
melakukan mapping dokumen dan term ke sebuah integer. Ingat bahwa dengan
skema BSBI, kita perlu mengelola struktur data untuk _mapping_ Term-Term ID. Di file
**util.py** , ada juga fungsi yang berguna untuk melakukan _merging_ atau penggabungan
hasil dari dua buah _postings lists_.

File **compression.py** berisi kebutuhan untuk melakukan _index compression_. Saat ini,
baru ada metode **Variable Byte Encoding** (VBE).

File **search.py** memberikan contoh bagaimana melakukan proses pencarian diberikan
beberapa **query**. Perhatikan bahwa inverted index terlebih dahulu, sebelum **search.py**
dieksekusi.

File **evaluation.py** digunakan untuk melakukan evaluasi seberapa efektif search
engine Anda diberikan query relevance judgment ( **qrels.txt** ) dan daftar _query_ yang
cukup banyak (30 _query_ ) pada **queries.txt**. Di dalamnya juga ada implementasi sebuah
metric, yaitu **Rank-Biased Precision (RBP)**.

Cara menjalankan:

1. Jangan lupa melakukan indexing terlebih dahulu dengan melakukan run
    **bsbi.py**
2. Silakan coba melakan retrieval untuk beberapa query, dengan menjalankan
    **search.py**
3. Jika Anda penasaran dengan kualitas search engine Anda, silakan run
    **evaluation.py** dan lihat skor RBP-nya.


**Tugas Anda (100 Point):**

1. [ 3 0 Point] Tambahkan **satu** algoritma kompresi lain selain VBE, tetapi harus
    yang bersifat **Bit-Level** seperti Elias-Gamma, OptPForDelta, **atau** yang lainnya.
2. [30 Point] Tambahkan fungsionalitas bisa melakukan _scoring_ dengan **BM**.
    Saat ini, hanya bisa menggunakan TF-IDF biasa. Untuk menambahkan
    fungsionalitas **BM25** , Anda perlu melakukan pre-komputasi untuk panjang
    dokumen dan juga rataan panjang dokumen saat _indexing_ dan disimpan di
    suatu struktur data.
3. [30 Point] Tambahkan **tiga** metrik evaluasi lain: **NDCG** , **DCG** , dan **AP**.
4. [ 1 0 Point] Tambahkan algoritma **WAND Top-K Retrieval** agar tidak semua
    dokumen dihitung skor **BM25** - nya. Anda perlu edit dan sesuaikan isi dari
    _Inverted Index_ - nya.

**Bonus (150 Point)!**

Ya, jika Anda mengerjakan tugas utama (100 Point) dan mendapatkan nilai penuh
untuk Bonus (150 Point), Anda akan mendapatkan total 250 Point untuk Tugas
Pemrograman 2 ini. Perhatikan juga bahwa persentase Tugas Pemrograman 2 aslinya
adalah 4%, yaitu normalnya berkontribusi sebesar 4 Point untuk nilai akhir. Namun,
jika Anda mendapatkan full 250 Point, Anda akan memperoleh full 10 Point ke nilai
akhir kuliah TBI ini. **Jadi, jangan sampai Anda tidak memanfaatkan kesempatan ini.**

Bonus ini diberikan kepada peserta yang merasa tidak puas dengan Empat fitur
utama yang sudah dibahas sebelumnya, dan merasa ingin membuat _search engine_ yang
bisa mendekati _library search engine_ yang available di Internet, seperti:

- https://pyterrier.readthedocs.io/en/latest/
    o (https://github.com/terrier-org/pyterrier)
- https://github.com/castorini/pyserini
- Dsb.

Penilaian bonus akan didasarkan kepada usaha yang diberikan oleh peserta untuk
mengimplementasikan fitur-fitur tambahan lain, seperti:

1. Ada mode indexing dengan **SPIMI** , daripada **BSBI** ;
2. _Term-term_ pada _dictionary_ disusun menggunakan struktur data yang lebih
    kompleks dan efisien seperti **TRIE** , **Patricia Tree** , atau **Finite State Transducer**
    (FST);


3. Ada opsi untuk melakukan _Latent Semantic Indexing_ dengan vektor-vektor
    dokumen di-index menggunakan Teknik _Vector Indexing_ seperti **FAISS**
    (https://github.com/facebookresearch/faiss). Namun, Anda harus memikirkan
    bagaimana melakukan SVD yang efisien melibatkan _Term-Document Matrix_
    yang sangat besar. Terkait _Vector Indexing_ , Anda akan belajar di kelas nanti.
4. Adaptive Retrieval: https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-
    adaptive/index.html
5. Dan fitur-fitur menarik lainnya ...
