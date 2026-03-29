"""
FST-based Indexer Implementation

This module provides an indexer that uses Finite State Transducer (FST)
for the term dictionary instead of a simple hash map.

Advantages of FST-based dictionary:
1. Memory efficient for large vocabularies with shared prefixes/suffixes
2. Supports prefix-based queries (autocomplete, wildcard prefix)
3. Supports fuzzy search (spelling correction)
4. Ordered iteration of terms
"""

import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from fst import FST, FSTDictionary, MinimalFST
from tqdm import tqdm


class FSTIndex:
    """
    Inverted Index with FST-based term dictionary.

    FST digunakan untuk menyimpan mapping term -> term_id dengan fitur tambahan:
    - Pencarian prefix (untuk autocomplete)
    - Pencarian fuzzy (untuk spelling correction)
    - Hemat memori untuk vocabulary besar

    Attributes
    ----------
    term_fst: FSTDictionary
        FST-based dictionary untuk term -> term_id mapping
    doc_id_map: IdMap
        Mapping doc paths to doc IDs
    data_dir: str
        Path ke data
    output_dir: str
        Path ke output index files
    postings_encoding: encoding class
        Encoding untuk postings list
    index_name: str
        Nama index file
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index"):
        self.term_fst = FSTDictionary()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.intermediate_indices = []

    def save(self):
        """Save FST dictionary and doc_id_map"""
        # Save FST-based term dictionary
        self.term_fst.save(os.path.join(self.output_dir, 'terms_fst.dict'))

        # Save doc_id_map (standard pickle)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

        # Also save as standard IdMap for compatibility
        term_id_map = IdMap()
        term_id_map.str_to_id = {term: tid for term, tid in self.term_fst.iterate_all()}
        term_id_map.id_to_str = self.term_fst.id_to_str
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(term_id_map, f)

    def load(self):
        """Load FST dictionary and doc_id_map"""
        fst_path = os.path.join(self.output_dir, 'terms_fst.dict')
        if os.path.exists(fst_path):
            self.term_fst = FSTDictionary.load(fst_path)
        else:
            # Fallback to standard IdMap
            with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
                term_id_map = pickle.load(f)
            # Convert to FST
            self.term_fst = FSTDictionary()
            for term, tid in term_id_map.str_to_id.items():
                self.term_fst.fst.add(term, tid)
            self.term_fst.id_to_str = term_id_map.id_to_str

        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Parse documents dalam sebuah block dan kembalikan td_pairs.

        Parameters
        ----------
        block_dir_relative: str
            Relative path ke directory block

        Returns
        -------
        List[Tuple[int, int]]
            List of (term_id, doc_id) pairs
        """
        dir_path = os.path.join(".", self.data_dir, block_dir_relative)
        td_pairs = []

        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            if os.path.isfile(filepath):
                doc_id = self.doc_id_map[filepath]

                with open(filepath, "r", encoding="utf8", errors="surrogateescape") as f:
                    for token in f.read().split():
                        term_id = self.term_fst[token]
                        td_pairs.append((term_id, doc_id))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Invert td_pairs dan tulis ke index file.
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
        Merge intermediate indices ke merged_index.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)

        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_

        merged_index.append(curr, postings, tf_list)

    def index(self):
        """
        Build index dengan FST-based dictionary.
        """
        print("=" * 60)
        print("FST-BASED INDEXING")
        print("=" * 60)

        # Process each block
        block_dirs = sorted(next(os.walk(self.data_dir))[1])

        for block_dir in tqdm(block_dirs, desc="Processing blocks"):
            td_pairs = self.parse_block(block_dir)
            index_id = f'fst_intermediate_{block_dir}'
            self.intermediate_indices.append(index_id)

            with InvertedIndexWriter(index_id, self.postings_encoding,
                                     directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)

        self.save()

        # Merge intermediate indices
        print("Merging intermediate indices...")
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(index_id, self.postings_encoding,
                                           directory=self.output_dir)
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)

        # Cleanup intermediate files
        for index_id in self.intermediate_indices:
            index_path = os.path.join(self.output_dir, f'{index_id}.index')
            dict_path = os.path.join(self.output_dir, f'{index_id}.dict')
            for path in [index_path, dict_path]:
                if os.path.exists(path):
                    os.remove(path)

        # Print FST statistics
        stats = self.term_fst.get_stats()
        print(f"\nIndexing complete!")
        print(f"Total terms: {stats['total_terms']}")
        print(f"FST nodes: {stats['node_count']}")
        print(f"FST edges: {stats['edge_count']}")
        print(f"Total documents: {len(self.doc_id_map)}")

    def retrieve_tfidf(self, query, k=10):
        """
        TF-IDF retrieval.
        """
        if len(self.term_fst) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in query.split():
            term_id = self.term_fst.get_id(word)
            if term_id is not None:
                terms.append(term_id)

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

            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        BM25 retrieval.
        """
        if len(self.term_fst) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in query.split():
            term_id = self.term_fst.get_id(word)
            if term_id is not None:
                terms.append(term_id)

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

            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    # ==== FST-specific features ====

    def prefix_search(self, prefix: str, k: int = 10):
        """
        Cari terms yang dimulai dengan prefix tertentu.

        Fitur ini hanya tersedia karena menggunakan FST!

        Parameters
        ----------
        prefix: str
            Prefix yang dicari
        k: int
            Jumlah maksimum terms yang dikembalikan

        Returns
        -------
        List[Tuple[str, int]]
            List of (term, term_id) yang dimulai dengan prefix
        """
        if len(self.term_fst) == 0:
            self.load()

        results = self.term_fst.prefix_search(prefix)
        return results[:k]

    def fuzzy_term_search(self, term: str, max_distance: int = 2):
        """
        Cari terms yang mirip dengan term input (spelling correction).

        Fitur ini hanya tersedia karena menggunakan FST!

        Parameters
        ----------
        term: str
            Term yang dicari
        max_distance: int
            Maksimum edit distance

        Returns
        -------
        List[Tuple[str, int, int]]
            List of (term, term_id, edit_distance)
        """
        if len(self.term_fst) == 0:
            self.load()

        return self.term_fst.fuzzy_search(term, max_distance)

    def query_expansion_with_prefix(self, query: str, k: int = 10):
        """
        Expand query dengan menambahkan terms yang berbagi prefix.

        Contoh:
        - "comput" -> ["computer", "computing", "computation", ...]

        Parameters
        ----------
        query: str
            Query asli
        k: int
            Jumlah dokumen yang dikembalikan

        Returns
        -------
        List[(float, str)]
            Hasil retrieval dengan query expansion
        """
        if len(self.term_fst) == 0:
            self.load()

        expanded_terms = set()

        # Untuk setiap word dalam query
        for word in query.split():
            expanded_terms.add(word)
            # Tambahkan terms dengan prefix sama
            prefix_results = self.term_fst.prefix_search(word[:3])  # 3 char prefix
            for term, _ in prefix_results[:5]:  # Limit 5 per word
                expanded_terms.add(term)

        # Run BM25 dengan expanded query
        expanded_query = ' '.join(expanded_terms)
        return self.retrieve_bm25(expanded_query, k=k)

    def spell_corrected_search(self, query: str, k: int = 10, max_distance: int = 1):
        """
        Search dengan spell correction otomatis.

        Jika term tidak ditemukan di vocabulary, cari term yang mirip.

        Parameters
        ----------
        query: str
            Query asli
        k: int
            Jumlah dokumen yang dikembalikan
        max_distance: int
            Maksimum edit distance untuk spell correction

        Returns
        -------
        Tuple[List[(float, str)], Dict[str, str]]
            (hasil retrieval, corrections dict)
        """
        if len(self.term_fst) == 0:
            self.load()

        corrected_terms = []
        corrections = {}

        for word in query.split():
            term_id = self.term_fst.get_id(word)

            if term_id is not None:
                corrected_terms.append(word)
            else:
                # Term tidak ditemukan, cari yang mirip
                similar = self.fuzzy_term_search(word, max_distance)
                if similar:
                    best_match = similar[0][0]  # Term dengan distance terkecil
                    corrected_terms.append(best_match)
                    corrections[word] = best_match
                else:
                    # Tidak ada yang mirip, keep original
                    corrected_terms.append(word)

        corrected_query = ' '.join(corrected_terms)
        results = self.retrieve_bm25(corrected_query, k=k)

        return results, corrections


if __name__ == "__main__":
    # Test FST-based indexing
    print("=" * 60)
    print("FST INDEX TEST")
    print("=" * 60)

    fst_index = FSTIndex(
        data_dir='collection',
        postings_encoding=VBEPostings,
        output_dir='index'
    )

    fst_index.index()

    # Test retrieval
    print("\n--- Retrieval Tests ---")
    queries = [
        "alkylated with radioactive iodoacetate",
        "lipid metabolism"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("BM25 Results:")
        for score, doc in fst_index.retrieve_bm25(query, k=3):
            print(f"  {doc}: {score:.3f}")

    # Test FST-specific features
    print("\n--- FST-Specific Features ---")

    # Prefix search
    print("\nPrefix search 'meta':")
    for term, tid in fst_index.prefix_search("meta", k=10):
        print(f"  {term} (id={tid})")

    # Fuzzy search
    print("\nFuzzy search 'protien' (typo for 'protein'):")
    for term, tid, dist in fst_index.fuzzy_term_search("protien", max_distance=2):
        print(f"  {term} (id={tid}, distance={dist})")

    # Spell-corrected search
    print("\nSpell-corrected search:")
    query = "protien metablism"  # Typos
    results, corrections = fst_index.spell_corrected_search(query, k=3)
    print(f"  Original: {query}")
    print(f"  Corrections: {corrections}")
    print(f"  Results:")
    for score, doc in results:
        print(f"    {doc}: {score:.3f}")
