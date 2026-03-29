"""
SPIMI (Single-Pass In-Memory Indexing) Implementation

SPIMI is different from BSBI in several key ways:
1. BSBI uses term IDs (integers) during indexing, requiring a global term-to-termID mapping
2. SPIMI uses term strings directly during indexing, building the dictionary on-the-fly
3. SPIMI is more memory efficient for large vocabularies

Key characteristics of SPIMI:
- Processes terms as strings directly without converting to term IDs first
- Accumulates postings in memory using a dictionary (hashtable)
- When memory is full, sorts terms and writes to disk
- Term IDs are only assigned after the final merge
"""

import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm


class SPIMIIndex:
    """
    SPIMI (Single-Pass In-Memory Indexing) Implementation

    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs (dibuat setelah indexing selesai)
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen ke docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Encoding untuk postings (StandardPostings, VBEPostings, dll)
    index_name(str): Nama dari file yang berisi inverted index
    memory_threshold(int): Jumlah maksimum term-posting pairs sebelum flush ke disk
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", memory_threshold=100000):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.memory_threshold = memory_threshold

        # In-memory dictionary: term_string -> {doc_id -> tf}
        self.in_memory_index = {}
        self.current_size = 0

        # List of intermediate index files
        self.intermediate_indices = []
        self.block_counter = 0

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

    def add_to_index(self, term, doc_id):
        """
        Menambahkan term-doc pair ke in-memory index.

        Perbedaan utama dengan BSBI:
        - BSBI: Mengubah term ke term_id terlebih dahulu, lalu menyimpan (term_id, doc_id) pairs
        - SPIMI: Menyimpan term sebagai string langsung dalam dictionary

        Parameters
        ----------
        term: str
            Term string (bukan term ID)
        doc_id: int
            Document ID
        """
        if term not in self.in_memory_index:
            self.in_memory_index[term] = {}

        if doc_id not in self.in_memory_index[term]:
            self.in_memory_index[term][doc_id] = 0
            self.current_size += 1

        self.in_memory_index[term][doc_id] += 1

        # Check if we need to flush to disk
        if self.current_size >= self.memory_threshold:
            self.flush_to_disk()

    def flush_to_disk(self):
        """
        Menulis in-memory index ke disk.

        Perbedaan dengan BSBI:
        - BSBI: Sudah menggunakan term_ids, langsung sort berdasarkan term_id
        - SPIMI: Sort berdasarkan term string, term_id baru di-assign saat merge

        Format intermediate index untuk SPIMI:
        - Menggunakan InvertedIndexWriter yang sudah ada
        - Tapi term yang disimpan adalah hasil dari term_id_map[term_string]
        - Setiap block punya term_id_map sendiri (lokal)
        """
        if not self.in_memory_index:
            return

        # Create local term_id_map for this block
        local_term_map = IdMap()

        # Sort terms alphabetically
        sorted_terms = sorted(self.in_memory_index.keys())

        index_id = f'spimi_intermediate_{self.block_counter}'
        self.intermediate_indices.append(index_id)

        with InvertedIndexWriter(index_id, self.postings_encoding,
                                 directory=self.output_dir) as index:
            for term in sorted_terms:
                term_id = local_term_map[term]
                doc_tfs = self.in_memory_index[term]

                # Sort by doc_id
                sorted_doc_ids = sorted(doc_tfs.keys())
                tf_list = [doc_tfs[doc_id] for doc_id in sorted_doc_ids]

                index.append(term_id, sorted_doc_ids, tf_list)

        # Save local term map for this block
        term_map_path = os.path.join(self.output_dir, f'{index_id}_terms.pkl')
        with open(term_map_path, 'wb') as f:
            pickle.dump(local_term_map, f)

        # Clear in-memory index
        self.in_memory_index = {}
        self.current_size = 0
        self.block_counter += 1

    def parse_and_index_block(self, block_dir_relative):
        """
        Parse dokumen dan langsung tambahkan ke in-memory index.

        Perbedaan dengan BSBI:
        - BSBI (parse_block): Menghasilkan list of (term_id, doc_id) pairs
        - SPIMI: Langsung memasukkan term ke dictionary tanpa konversi ke term_id

        Parameters
        ----------
        block_dir_relative: str
            Relative path ke directory yang mengandung text files
        """
        dir_path = os.path.join(".", self.data_dir, block_dir_relative)

        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            if os.path.isfile(filepath):
                doc_id = self.doc_id_map[filepath]

                with open(filepath, "r", encoding="utf8", errors="surrogateescape") as f:
                    for token in f.read().split():
                        self.add_to_index(token, doc_id)

    def merge_spimi_indices(self, merged_index):
        """
        Merge semua intermediate SPIMI indices menjadi satu final index.

        Perbedaan dengan BSBI merge:
        - BSBI: Semua block menggunakan global term_id_map yang sama
        - SPIMI: Setiap block punya local term_map, perlu di-merge berdasarkan term string

        Proses:
        1. Load semua intermediate indices dan local term maps
        2. Iterasi berdasarkan term string (bukan term_id)
        3. Assign global term_id saat menulis ke final index
        """
        # Load all intermediate indices and their term maps
        index_data = []

        for index_id in self.intermediate_indices:
            # Load local term map
            term_map_path = os.path.join(self.output_dir, f'{index_id}_terms.pkl')
            with open(term_map_path, 'rb') as f:
                local_term_map = pickle.load(f)

            # Create reverse mapping: term_id -> term_string
            id_to_term = {i: local_term_map[i] for i in range(len(local_term_map))}

            index_data.append({
                'index_id': index_id,
                'local_term_map': local_term_map,
                'id_to_term': id_to_term
            })

        # Use heap to merge indices by term string
        # Each heap element: (term_string, block_idx, term_id_in_block, postings, tf_list)
        heap = []
        index_readers = []
        index_iters = []

        with contextlib.ExitStack() as stack:
            # Open all intermediate indices
            for i, data in enumerate(index_data):
                reader = stack.enter_context(
                    InvertedIndexReader(data['index_id'], self.postings_encoding,
                                       directory=self.output_dir)
                )
                index_readers.append(reader)
                index_iters.append(iter(reader))

                # Add first element from each index to heap
                try:
                    term_id, postings, tf_list = next(index_iters[i])
                    term_string = data['id_to_term'][term_id]
                    heapq.heappush(heap, (term_string, i, postings, tf_list))
                except StopIteration:
                    pass

            # Merge process
            current_term = None
            current_postings = []
            current_tfs = []

            while heap:
                term_string, block_idx, postings, tf_list = heapq.heappop(heap)

                if current_term is None:
                    current_term = term_string
                    current_postings = list(postings)
                    current_tfs = list(tf_list)
                elif term_string == current_term:
                    # Merge postings for the same term
                    merged = sorted_merge_posts_and_tfs(
                        list(zip(current_postings, current_tfs)),
                        list(zip(postings, tf_list))
                    )
                    current_postings = [p for p, _ in merged]
                    current_tfs = [tf for _, tf in merged]
                else:
                    # Write previous term to final index
                    global_term_id = self.term_id_map[current_term]
                    merged_index.append(global_term_id, current_postings, current_tfs)

                    # Start new term
                    current_term = term_string
                    current_postings = list(postings)
                    current_tfs = list(tf_list)

                # Add next element from the same block
                try:
                    term_id, postings, tf_list = next(index_iters[block_idx])
                    term_string = index_data[block_idx]['id_to_term'][term_id]
                    heapq.heappush(heap, (term_string, block_idx, postings, tf_list))
                except StopIteration:
                    pass

            # Don't forget the last term
            if current_term is not None:
                global_term_id = self.term_id_map[current_term]
                merged_index.append(global_term_id, current_postings, current_tfs)

    def cleanup_intermediate_files(self):
        """Hapus file intermediate index setelah merge selesai"""
        for index_id in self.intermediate_indices:
            # Remove index file
            index_path = os.path.join(self.output_dir, f'{index_id}.index')
            dict_path = os.path.join(self.output_dir, f'{index_id}.dict')
            term_map_path = os.path.join(self.output_dir, f'{index_id}_terms.pkl')

            for path in [index_path, dict_path, term_map_path]:
                if os.path.exists(path):
                    os.remove(path)

    def index(self):
        """
        SPIMI Indexing - Main method

        Perbedaan utama dengan BSBI:

        BSBI:
        1. parse_block: Tokenize dan konversi ke (term_id, doc_id) pairs
        2. invert_write: Sort pairs dan tulis ke intermediate index
        3. merge: Gabungkan berdasarkan term_id

        SPIMI:
        1. parse_and_index_block: Tokenize dan langsung masukkan ke in-memory dictionary
        2. flush_to_disk: Ketika memory penuh, sort dan tulis ke disk
        3. merge_spimi_indices: Gabungkan berdasarkan term string, assign global term_id
        """
        print("=" * 60)
        print("SPIMI INDEXING")
        print("=" * 60)
        print(f"Memory threshold: {self.memory_threshold} term-doc pairs")
        print()

        # Process all blocks
        block_dirs = sorted(next(os.walk(self.data_dir))[1])

        for block_dir in tqdm(block_dirs, desc="Processing blocks"):
            self.parse_and_index_block(block_dir)

        # Flush any remaining data
        self.flush_to_disk()

        print(f"\nCreated {len(self.intermediate_indices)} intermediate indices")
        print("Merging intermediate indices...")

        # Save doc_id_map (term_id_map will be built during merge)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

        # Merge all intermediate indices
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            self.merge_spimi_indices(merged_index)

        # Save final term_id_map
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)

        # Cleanup intermediate files
        self.cleanup_intermediate_files()

        print(f"Indexing complete!")
        print(f"Total terms: {len(self.term_id_map)}")
        print(f"Total documents: {len(self.doc_id_map)}")

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).

        Sama dengan BSBI karena final index memiliki format yang sama.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
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

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema BM25.

        Sama dengan BSBI karena final index memiliki format yang sama.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []

            avgdl = sum(merged_index.doc_length.values()) / N
            scores = {}

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term)

                    for doc_id, tf in zip(postings, tf_list):
                        if doc_id not in scores:
                            scores[doc_id] = 0

                        doc_len = merged_index.doc_length.get(doc_id, avgdl)
                        numerator = (k1 + 1) * tf
                        denominator = k1 * (1 - b + b * (doc_len / avgdl)) + tf
                        scores[doc_id] += idf * (numerator / denominator)

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]


if __name__ == "__main__":
    # Test SPIMI indexing
    SPIMI_instance = SPIMIIndex(
        data_dir='collection',
        postings_encoding=VBEPostings,
        output_dir='index',
        memory_threshold=50000  # Smaller threshold for testing
    )
    SPIMI_instance.index()

    # Test retrieval
    print("\n" + "=" * 60)
    print("Testing SPIMI Retrieval")
    print("=" * 60)

    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("TF-IDF Results:")
        for score, doc in SPIMI_instance.retrieve_tfidf(query, k=3):
            print(f"  {doc}: {score:.3f}")

        print("BM25 Results:")
        for score, doc in SPIMI_instance.retrieve_bm25(query, k=3):
            print(f"  {doc}: {score:.3f}")
