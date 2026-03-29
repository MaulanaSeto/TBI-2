import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema BM25 (Best Matching 25).
        Method akan mengembalikan top-K retrieval results.

        Formula BM25:
        score(Q, D) = Σ IDF(t) × [TF(t,D) × (k1+1)] / [TF(t,D) + k1 × (1-b + b×|D|/avgdl)]

        Dimana:
        - IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        - |D| = panjang dokumen (jumlah token)
        - avgdl = rata-rata panjang dokumen dalam koleksi
        - k1 = parameter saturasi term frequency (default: 1.5)
        - b = parameter normalisasi panjang dokumen (default: 0.75)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah dokumen yang dikembalikan (default: 10)
        k1: float
            Parameter saturasi TF (default: 1.5)
            Nilai lebih tinggi = pengaruh TF lebih besar
        b: float
            Parameter normalisasi panjang dokumen (default: 0.75)
            b=0: tidak ada normalisasi panjang
            b=1: normalisasi penuh

        Result
        ------
        List[(float, str)]
            List of tuple: elemen pertama adalah score BM25, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            # Hitung average document length
            N = len(merged_index.doc_length)
            if N == 0:
                return []

            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]

                    # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                    postings, tf_list = merged_index.get_postings_list(term)

                    for doc_id, tf in zip(postings, tf_list):
                        if doc_id not in scores:
                            scores[doc_id] = 0

                        # Panjang dokumen
                        doc_len = merged_index.doc_length.get(doc_id, avgdl)

                        # BM25 TF component dengan length normalization
                        # numerator = (k1 + 1) * tf
                        # denominator = k1 * (1 - b + b * (doc_len / avgdl)) + tf
                        numerator = (k1 + 1) * tf
                        denominator = k1 * (1 - b + b * (doc_len / avgdl)) + tf

                        scores[doc_id] += idf * (numerator / denominator)

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan BM25 menggunakan algoritma WAND
        (Weak AND) untuk optimasi Top-K retrieval.

        WAND adalah algoritma yang menghindari scoring semua dokumen dengan
        menggunakan upper bound pada kontribusi setiap term. Dokumen yang
        tidak mungkin masuk top-K akan di-skip.

        Cara kerja WAND:
        1. Pre-compute upper bound score untuk setiap term
        2. Maintain current top-K dengan threshold score minimum
        3. Untuk setiap dokumen, hitung upper bound total score
        4. Skip dokumen jika upper bound < threshold
        5. Full evaluate hanya dokumen yang berpotensi masuk top-K

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah dokumen yang dikembalikan (default: 10)
        k1: float
            Parameter BM25 k1 (default: 1.5)
        b: float
            Parameter BM25 b (default: 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple: (score BM25, nama dokumen)
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_terms = query.split()
        terms = [self.term_id_map[word] for word in query_terms]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []

            avgdl = sum(merged_index.doc_length.values()) / N

            # Step 1: Pre-compute upper bound untuk setiap term
            # Upper bound = IDF * max possible TF component
            # Max TF component terjadi ketika TF sangat besar dan doc_len = avgdl
            # Dalam kasus tersebut: (k1 + 1) * tf / (k1 + tf) -> mendekati (k1 + 1)

            term_data = {}  # term -> (idf, upper_bound, postings, tf_list)

            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                postings, tf_list = merged_index.get_postings_list(term)

                # Upper bound: asumsi dokumen dengan TF tinggi dan panjang = avgdl
                # Saat b=0.75 dan doc_len=avgdl: denominator = k1 * (1-b+b) + tf = k1 + tf
                # Jadi TF component = (k1+1)*tf / (k1+tf)
                # Maksimum mendekati (k1+1) saat tf -> infinity

                # Untuk upper bound yang lebih tight, gunakan max TF dalam postings
                if tf_list:
                    max_tf = max(tf_list)
                    # Asumsi dokumen terpendek (length normalization paling menguntungkan)
                    min_doc_len = min(merged_index.doc_length.get(p, avgdl) for p in postings)
                    norm_factor = 1 - b + b * (min_doc_len / avgdl)
                    upper_bound = idf * ((k1 + 1) * max_tf) / (k1 * norm_factor + max_tf)
                else:
                    upper_bound = idf * (k1 + 1)

                term_data[term] = {
                    'idf': idf,
                    'upper_bound': upper_bound,
                    'postings': postings,
                    'tf_list': tf_list,
                    'posting_idx': 0  # Current position in postings list
                }

            if not term_data:
                return []

            # Step 2: Collect all unique doc_ids dan their potential scores
            # Untuk WAND yang lebih efisien, kita menggunakan document-at-a-time
            # dengan pivot selection, tapi di sini kita gunakan versi simplified

            # Collect all documents yang muncul di setidaknya satu postings list
            doc_scores = {}
            doc_upper_bounds = {}

            for term, data in term_data.items():
                for doc_id in data['postings']:
                    if doc_id not in doc_upper_bounds:
                        doc_upper_bounds[doc_id] = 0
                    doc_upper_bounds[doc_id] += data['upper_bound']

            # Step 3: Gunakan heap untuk maintain top-K
            # Hanya evaluate dokumen yang upper bound >= current threshold
            top_k_heap = []  # min-heap of (score, doc_id)
            threshold = 0.0

            # Sort documents by upper bound (descending) untuk early termination
            sorted_docs = sorted(doc_upper_bounds.items(),
                                key=lambda x: x[1], reverse=True)

            # Step 4: Evaluate dokumen dengan WAND pruning
            docs_evaluated = 0
            docs_pruned = 0

            for doc_id, upper_bound in sorted_docs:
                # WAND pruning: skip jika upper bound < threshold
                if upper_bound < threshold:
                    docs_pruned += 1
                    continue

                docs_evaluated += 1

                # Full evaluation untuk dokumen ini
                score = 0.0
                doc_len = merged_index.doc_length.get(doc_id, avgdl)

                for term, data in term_data.items():
                    # Binary search untuk cek apakah doc_id ada di postings
                    postings = data['postings']
                    tf_list = data['tf_list']

                    # Linear search (bisa dioptimasi dengan binary search)
                    try:
                        idx = postings.index(doc_id)
                        tf = tf_list[idx]

                        idf = data['idf']
                        numerator = (k1 + 1) * tf
                        denominator = k1 * (1 - b + b * (doc_len / avgdl)) + tf
                        score += idf * (numerator / denominator)
                    except ValueError:
                        # doc_id tidak ada di postings untuk term ini
                        pass

                # Update top-K heap
                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (score, doc_id))
                    if len(top_k_heap) == k:
                        threshold = top_k_heap[0][0]  # Update threshold
                elif score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (score, doc_id))
                    threshold = top_k_heap[0][0]  # Update threshold

            # Convert heap to sorted result
            results = [(score, self.doc_id_map[doc_id])
                      for score, doc_id in top_k_heap]

            return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
