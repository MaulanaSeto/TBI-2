"""
Latent Semantic Indexing (LSI) Implementation

LSI (Latent Semantic Indexing) adalah teknik yang menggunakan SVD (Singular Value
Decomposition) untuk menemukan struktur semantik laten dalam koleksi dokumen.

Keunggulan LSI:
1. Menangani sinonimi: kata berbeda dengan makna sama akan memiliki representasi mirip
2. Menangani polisemi: dapat membedakan makna kata berdasarkan konteks
3. Dimensionality reduction: representasi dokumen lebih compact

Proses LSI:
1. Build Term-Document Matrix (TDM) dengan TF-IDF weighting
2. Apply Truncated SVD: A ≈ U_k * Σ_k * V_k^T
3. Document vectors = rows of V_k * Σ_k (atau U_k * Σ_k untuk term vectors)
4. Query projection: q' = q^T * U_k * Σ_k^(-1)
5. Similarity search menggunakan cosine similarity
"""

import os
import pickle
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import heapq

# Try to import numpy for efficient matrix operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Using pure Python implementation (slower).")


class SparseMatrix:
    """
    Sparse matrix implementation untuk Term-Document Matrix.
    Menggunakan format COO (Coordinate List) yang efisien untuk konstruksi.
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = defaultdict(float)  # (row, col) -> value

    def __setitem__(self, key: Tuple[int, int], value: float):
        row, col = key
        if value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]

    def __getitem__(self, key: Tuple[int, int]) -> float:
        return self.data.get(key, 0.0)

    def to_dense(self) -> List[List[float]]:
        """Convert to dense matrix (list of lists)"""
        matrix = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        for (row, col), value in self.data.items():
            matrix[row][col] = value
        return matrix

    def to_numpy(self):
        """Convert to numpy array if available"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy not available")
        matrix = np.zeros((self.n_rows, self.n_cols))
        for (row, col), value in self.data.items():
            matrix[row, col] = value
        return matrix

    def get_row(self, row: int) -> Dict[int, float]:
        """Get non-zero elements in a row"""
        return {col: val for (r, col), val in self.data.items() if r == row}

    def get_col(self, col: int) -> Dict[int, float]:
        """Get non-zero elements in a column"""
        return {row: val for (row, c), val in self.data.items() if c == col}


class VectorIndex:
    """
    Simple Vector Index untuk similarity search.

    Implementasi sederhana yang mirip dengan FAISS untuk:
    - Indexing document vectors
    - Fast similarity search (brute force dengan optimisasi)

    Untuk production, gunakan FAISS atau Annoy untuk approximate nearest neighbor.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []  # List of (doc_id, vector)
        self.norms = []    # Pre-computed norms for faster cosine similarity

    def add(self, doc_id: int, vector: List[float]):
        """Add a document vector to the index"""
        norm = math.sqrt(sum(v * v for v in vector))
        self.vectors.append((doc_id, vector))
        self.norms.append(norm)

    def add_batch(self, doc_vectors: List[Tuple[int, List[float]]]):
        """Add multiple document vectors"""
        for doc_id, vector in doc_vectors:
            self.add(doc_id, vector)

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for k most similar documents.

        Returns list of (doc_id, similarity_score) sorted by score descending.
        """
        query_norm = math.sqrt(sum(v * v for v in query_vector))

        if query_norm == 0:
            return []

        scores = []
        for i, (doc_id, doc_vector) in enumerate(self.vectors):
            doc_norm = self.norms[i]
            if doc_norm == 0:
                continue

            # Cosine similarity
            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            similarity = dot_product / (query_norm * doc_norm)
            scores.append((doc_id, similarity))

        # Get top-k
        return heapq.nlargest(k, scores, key=lambda x: x[1])

    def save(self, filepath: str):
        """Save index to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'vectors': self.vectors,
                'norms': self.norms
            }, f)

    @staticmethod
    def load(filepath: str) -> 'VectorIndex':
        """Load index from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        index = VectorIndex(data['dimension'])
        index.vectors = data['vectors']
        index.norms = data['norms']
        return index


class NumpyVectorIndex:
    """
    Numpy-optimized Vector Index untuk similarity search.
    Jauh lebih cepat dari pure Python implementation.
    """

    def __init__(self, dimension: int):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy required for NumpyVectorIndex")
        self.dimension = dimension
        self.doc_ids = []
        self.vectors = None  # Will be numpy array
        self._vectors_list = []  # Temporary storage during construction

    def add(self, doc_id: int, vector):
        """Add a document vector"""
        self.doc_ids.append(doc_id)
        self._vectors_list.append(vector)

    def build(self):
        """Build the index (convert to numpy array)"""
        if self._vectors_list:
            self.vectors = np.array(self._vectors_list)
            # Normalize vectors
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self.vectors = self.vectors / norms
            self._vectors_list = []  # Clear temporary storage

    def search(self, query_vector, k: int = 10) -> List[Tuple[int, float]]:
        """Search for k most similar documents using cosine similarity"""
        if self.vectors is None or len(self.vectors) == 0:
            return []

        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        # Compute all similarities at once
        similarities = self.vectors @ query

        # Get top-k indices
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self.doc_ids[i], float(similarities[i])) for i in top_indices[:k]]

    def save(self, filepath: str):
        """Save index to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'doc_ids': self.doc_ids,
                'vectors': self.vectors
            }, f)

    @staticmethod
    def load(filepath: str) -> 'NumpyVectorIndex':
        """Load index from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        index = NumpyVectorIndex(data['dimension'])
        index.doc_ids = data['doc_ids']
        index.vectors = data['vectors']
        return index


def truncated_svd_power_iteration(matrix: List[List[float]], n_components: int,
                                   n_iterations: int = 100) -> Tuple:
    """
    Truncated SVD menggunakan Power Iteration method.
    Pure Python implementation untuk kasus tanpa numpy.

    Returns: (U, S, Vt) dimana:
    - U: left singular vectors (m x n_components)
    - S: singular values (n_components,)
    - Vt: right singular vectors transposed (n_components x n)
    """
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    def mat_vec_mult(M, v):
        """Matrix-vector multiplication"""
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

    def vec_mat_mult(v, M):
        """Vector-matrix multiplication (v^T * M)"""
        return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(len(M[0]))]

    def normalize(v):
        """Normalize vector"""
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v, 0
        return [x / norm for x in v], norm

    def transpose(M):
        """Transpose matrix"""
        return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

    U = []
    S = []
    Vt = []

    # Deflation method: compute one singular value/vector at a time
    A = [row[:] for row in matrix]  # Copy

    for _ in range(min(n_components, min(m, n))):
        # Initialize random vector
        import random
        v = [random.gauss(0, 1) for _ in range(n)]
        v, _ = normalize(v)

        # Power iteration
        for _ in range(n_iterations):
            # u = A * v
            u = mat_vec_mult(A, v)
            u, _ = normalize(u)

            # v = A^T * u
            At = transpose(A)
            v = mat_vec_mult(At, u)
            v, sigma = normalize(v)

        if sigma < 1e-10:
            break

        U.append(u)
        S.append(sigma)
        Vt.append(v)

        # Deflate: A = A - sigma * u * v^T
        for i in range(m):
            for j in range(n):
                A[i][j] -= sigma * u[i] * v[j]

    return U, S, Vt


class LSIIndex:
    """
    Latent Semantic Indexing untuk search engine.

    Menggunakan SVD untuk dimensionality reduction dan
    vector similarity untuk retrieval.
    """

    def __init__(self, data_dir: str, output_dir: str, n_components: int = 100):
        """
        Parameters
        ----------
        data_dir: str
            Path ke folder collection
        output_dir: str
            Path ke folder output
        n_components: int
            Jumlah dimensi LSI (default: 100)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_components = n_components

        # Mappings
        self.term_to_id = {}
        self.id_to_term = []
        self.doc_to_id = {}
        self.id_to_doc = []

        # LSI components
        self.U = None  # Term vectors (term x component)
        self.S = None  # Singular values
        self.Vt = None  # Document vectors transposed (component x doc)

        # Vector index for fast retrieval
        self.vector_index = None

        # IDF values for query weighting
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()

    def build_term_document_matrix(self) -> SparseMatrix:
        """
        Build TF-IDF weighted Term-Document Matrix.

        Returns sparse matrix where:
        - Rows = terms
        - Columns = documents
        - Values = TF-IDF weights
        """
        print("Building Term-Document Matrix...")

        # First pass: collect all terms and documents
        doc_term_freq = {}  # doc_id -> {term -> freq}
        term_doc_freq = defaultdict(int)  # term -> number of docs containing term

        for block_dir in sorted(os.listdir(self.data_dir)):
            block_path = os.path.join(self.data_dir, block_dir)
            if not os.path.isdir(block_path):
                continue

            for filename in os.listdir(block_path):
                filepath = os.path.join(block_path, filename)
                if not os.path.isfile(filepath):
                    continue

                # Get document ID
                if filepath not in self.doc_to_id:
                    doc_id = len(self.id_to_doc)
                    self.doc_to_id[filepath] = doc_id
                    self.id_to_doc.append(filepath)
                else:
                    doc_id = self.doc_to_id[filepath]

                # Read and tokenize
                with open(filepath, 'r', encoding='utf8', errors='surrogateescape') as f:
                    tokens = self._tokenize(f.read())

                # Count term frequencies
                term_freq = defaultdict(int)
                for token in tokens:
                    if token not in self.term_to_id:
                        term_id = len(self.id_to_term)
                        self.term_to_id[token] = term_id
                        self.id_to_term.append(token)
                    term_freq[token] += 1

                doc_term_freq[doc_id] = dict(term_freq)

                # Update document frequencies
                for term in term_freq:
                    term_doc_freq[term] += 1

        n_terms = len(self.id_to_term)
        n_docs = len(self.id_to_doc)
        print(f"  Terms: {n_terms}, Documents: {n_docs}")

        # Compute IDF
        for term, df in term_doc_freq.items():
            self.idf[term] = math.log(n_docs / df)

        # Build sparse TF-IDF matrix
        matrix = SparseMatrix(n_terms, n_docs)

        for doc_id, term_freqs in doc_term_freq.items():
            # Compute document length for normalization
            doc_length = sum(term_freqs.values())

            for term, tf in term_freqs.items():
                term_id = self.term_to_id[term]
                # TF-IDF with log normalization
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                idf_weight = self.idf[term]
                matrix[term_id, doc_id] = tf_weight * idf_weight

        return matrix

    def compute_svd(self, matrix: SparseMatrix):
        """
        Compute truncated SVD of the term-document matrix.

        Untuk matrix besar, gunakan iterative methods yang lebih efisien.
        """
        print(f"Computing SVD with {self.n_components} components...")

        if NUMPY_AVAILABLE:
            # Use numpy's efficient SVD
            dense_matrix = matrix.to_numpy()

            # Truncated SVD using randomized method for efficiency
            from numpy.linalg import svd

            # For very large matrices, use randomized SVD
            if dense_matrix.shape[0] > 1000 or dense_matrix.shape[1] > 1000:
                print("  Using randomized SVD for large matrix...")
                U, S, Vt = self._randomized_svd(dense_matrix, self.n_components)
            else:
                U, S, Vt = svd(dense_matrix, full_matrices=False)
                # Truncate to n_components
                k = min(self.n_components, len(S))
                U = U[:, :k]
                S = S[:k]
                Vt = Vt[:k, :]

            self.U = U
            self.S = S
            self.Vt = Vt
        else:
            # Use pure Python power iteration
            print("  Using power iteration method (slower)...")
            dense_matrix = matrix.to_dense()
            U_list, S_list, Vt_list = truncated_svd_power_iteration(
                dense_matrix, self.n_components, n_iterations=50
            )

            # Convert to proper format
            self.U = [[U_list[j][i] for j in range(len(U_list))]
                      for i in range(len(U_list[0]))] if U_list else []
            self.S = S_list
            self.Vt = Vt_list

        print(f"  SVD complete. Retained {len(self.S)} components.")

    def _randomized_svd(self, matrix, n_components, n_oversamples=10, n_iter=5):
        """
        Randomized SVD for large matrices.
        More efficient than full SVD when n_components << min(m, n).
        """
        m, n = matrix.shape
        k = min(n_components + n_oversamples, min(m, n))

        # Random projection
        Q = np.random.randn(n, k)

        # Power iteration for better approximation
        for _ in range(n_iter):
            Q = matrix @ Q
            Q, _ = np.linalg.qr(Q)
            Q = matrix.T @ Q
            Q, _ = np.linalg.qr(Q)

        Q = matrix @ Q
        Q, _ = np.linalg.qr(Q)

        # SVD of smaller matrix
        B = Q.T @ matrix
        Uhat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat

        # Truncate
        return U[:, :n_components], S[:n_components], Vt[:n_components, :]

    def build_vector_index(self):
        """Build vector index untuk fast similarity search"""
        print("Building vector index...")

        if NUMPY_AVAILABLE:
            self.vector_index = NumpyVectorIndex(self.n_components)

            # Document vectors = V * S (atau columns of Vt.T * diag(S))
            # Each column of Vt is a document vector in the reduced space
            for doc_id in range(self.Vt.shape[1]):
                doc_vector = self.Vt[:, doc_id] * self.S
                self.vector_index.add(doc_id, doc_vector)

            self.vector_index.build()
        else:
            self.vector_index = VectorIndex(len(self.S))

            # Pure Python version
            for doc_id in range(len(self.Vt[0])):
                doc_vector = [self.Vt[k][doc_id] * self.S[k]
                              for k in range(len(self.S))]
                self.vector_index.add(doc_id, doc_vector)

        print(f"  Indexed {len(self.id_to_doc)} document vectors")

    def index(self):
        """Build the complete LSI index"""
        print("=" * 60)
        print("LSI INDEXING")
        print("=" * 60)

        # Build term-document matrix
        matrix = self.build_term_document_matrix()

        # Compute SVD
        self.compute_svd(matrix)

        # Build vector index
        self.build_vector_index()

        # Save index
        self.save()

        print("\nLSI indexing complete!")
        print(f"  Components: {len(self.S)}")
        print(f"  Documents: {len(self.id_to_doc)}")
        print(f"  Terms: {len(self.id_to_term)}")

    def save(self):
        """Save LSI index to disk"""
        index_path = os.path.join(self.output_dir, 'lsi_index.pkl')
        with open(index_path, 'wb') as f:
            pickle.dump({
                'n_components': self.n_components,
                'term_to_id': self.term_to_id,
                'id_to_term': self.id_to_term,
                'doc_to_id': self.doc_to_id,
                'id_to_doc': self.id_to_doc,
                'U': self.U if not NUMPY_AVAILABLE else self.U.tolist(),
                'S': self.S if not NUMPY_AVAILABLE else self.S.tolist(),
                'Vt': self.Vt if not NUMPY_AVAILABLE else self.Vt.tolist(),
                'idf': self.idf,
            }, f)

        # Save vector index separately
        vector_path = os.path.join(self.output_dir, 'lsi_vectors.pkl')
        self.vector_index.save(vector_path)

    def load(self):
        """Load LSI index from disk"""
        index_path = os.path.join(self.output_dir, 'lsi_index.pkl')
        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.n_components = data['n_components']
        self.term_to_id = data['term_to_id']
        self.id_to_term = data['id_to_term']
        self.doc_to_id = data['doc_to_id']
        self.id_to_doc = data['id_to_doc']
        self.idf = data['idf']

        if NUMPY_AVAILABLE:
            self.U = np.array(data['U'])
            self.S = np.array(data['S'])
            self.Vt = np.array(data['Vt'])
        else:
            self.U = data['U']
            self.S = data['S']
            self.Vt = data['Vt']

        # Load vector index
        vector_path = os.path.join(self.output_dir, 'lsi_vectors.pkl')
        if NUMPY_AVAILABLE:
            self.vector_index = NumpyVectorIndex.load(vector_path)
        else:
            self.vector_index = VectorIndex.load(vector_path)

    def _query_to_vector(self, query: str) -> List[float]:
        """
        Transform query ke LSI space.

        Query projection: q' = q^T * U * S^(-1)
        """
        tokens = self._tokenize(query)

        # Build query TF-IDF vector
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        if NUMPY_AVAILABLE:
            query_vector = np.zeros(len(self.id_to_term))
            for term, tf in term_freq.items():
                if term in self.term_to_id:
                    term_id = self.term_to_id[term]
                    tf_weight = 1 + math.log(tf) if tf > 0 else 0
                    idf_weight = self.idf.get(term, 0)
                    query_vector[term_id] = tf_weight * idf_weight

            # Project to LSI space: q' = q^T * U * S^(-1)
            # Equivalent to: q' = (U^T * q) / S
            projected = (self.U.T @ query_vector) / (self.S + 1e-10)
            return projected.tolist()
        else:
            # Pure Python version
            query_vec = [0.0] * len(self.id_to_term)
            for term, tf in term_freq.items():
                if term in self.term_to_id:
                    term_id = self.term_to_id[term]
                    tf_weight = 1 + math.log(tf) if tf > 0 else 0
                    idf_weight = self.idf.get(term, 0)
                    query_vec[term_id] = tf_weight * idf_weight

            # Project: q' = U^T * q / S
            projected = []
            for k in range(len(self.S)):
                val = sum(self.U[i][k] * query_vec[i] for i in range(len(query_vec)))
                projected.append(val / (self.S[k] + 1e-10))
            return projected

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[float, str]]:
        """
        Retrieve documents menggunakan LSI similarity.

        Parameters
        ----------
        query: str
            Query string
        k: int
            Number of results to return

        Returns
        -------
        List[(float, str)]
            List of (similarity_score, document_path)
        """
        if self.U is None:
            self.load()

        # Transform query to LSI space
        query_vector = self._query_to_vector(query)

        # Search in vector index
        results = self.vector_index.search(query_vector, k=k)

        # Convert to (score, doc_path) format
        return [(score, self.id_to_doc[doc_id]) for doc_id, score in results]

    def find_similar_documents(self, doc_path: str, k: int = 10) -> List[Tuple[float, str]]:
        """
        Find documents similar to a given document.

        Parameters
        ----------
        doc_path: str
            Path to the query document
        k: int
            Number of similar documents to return
        """
        if self.U is None:
            self.load()

        if doc_path not in self.doc_to_id:
            return []

        doc_id = self.doc_to_id[doc_path]

        # Get document vector
        if NUMPY_AVAILABLE:
            doc_vector = (self.Vt[:, doc_id] * self.S).tolist()
        else:
            doc_vector = [self.Vt[i][doc_id] * self.S[i] for i in range(len(self.S))]

        # Search (k+1 because the document itself will be in results)
        results = self.vector_index.search(doc_vector, k=k+1)

        # Filter out the query document itself
        return [(score, self.id_to_doc[did])
                for did, score in results if did != doc_id][:k]

    def get_term_similarity(self, term1: str, term2: str) -> float:
        """
        Compute semantic similarity between two terms.
        """
        if self.U is None:
            self.load()

        if term1 not in self.term_to_id or term2 not in self.term_to_id:
            return 0.0

        id1 = self.term_to_id[term1]
        id2 = self.term_to_id[term2]

        if NUMPY_AVAILABLE:
            vec1 = self.U[id1] * self.S
            vec2 = self.U[id2] * self.S
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        else:
            vec1 = [self.U[id1][k] * self.S[k] for k in range(len(self.S))]
            vec2 = [self.U[id2][k] * self.S[k] for k in range(len(self.S))]
            norm1 = math.sqrt(sum(v*v for v in vec1))
            norm2 = math.sqrt(sum(v*v for v in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return sum(v1*v2 for v1, v2 in zip(vec1, vec2)) / (norm1 * norm2)

    def find_similar_terms(self, term: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find terms semantically similar to the given term.
        """
        if self.U is None:
            self.load()

        if term not in self.term_to_id:
            return []

        term_id = self.term_to_id[term]

        # Get term vector
        if NUMPY_AVAILABLE:
            term_vector = self.U[term_id] * self.S

            # Compute similarities with all terms
            all_term_vectors = self.U * self.S
            norms = np.linalg.norm(all_term_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = all_term_vectors / norms

            query_norm = np.linalg.norm(term_vector)
            if query_norm == 0:
                return []
            query_normalized = term_vector / query_norm

            similarities = normalized @ query_normalized

            # Get top-k (excluding the term itself)
            top_indices = np.argsort(similarities)[::-1]
            results = []
            for idx in top_indices:
                if idx != term_id and len(results) < k:
                    results.append((self.id_to_term[idx], float(similarities[idx])))
            return results
        else:
            # Pure Python version (slower)
            term_vector = [self.U[term_id][i] * self.S[i] for i in range(len(self.S))]
            term_norm = math.sqrt(sum(v*v for v in term_vector))
            if term_norm == 0:
                return []

            similarities = []
            for other_id in range(len(self.id_to_term)):
                if other_id == term_id:
                    continue
                other_vector = [self.U[other_id][i] * self.S[i] for i in range(len(self.S))]
                other_norm = math.sqrt(sum(v*v for v in other_vector))
                if other_norm == 0:
                    continue
                sim = sum(t*o for t, o in zip(term_vector, other_vector)) / (term_norm * other_norm)
                similarities.append((self.id_to_term[other_id], sim))

            return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    # Test LSI indexing
    print("=" * 60)
    print("LSI INDEX TEST")
    print("=" * 60)

    lsi = LSIIndex(
        data_dir='collection',
        output_dir='index',
        n_components=50  # Smaller for testing
    )

    lsi.index()

    # Test retrieval
    print("\n--- Retrieval Test ---")
    queries = [
        "lipid metabolism",
        "protein synthesis",
        "cell division"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = lsi.retrieve(query, k=5)
        for score, doc in results:
            print(f"  {os.path.basename(doc):30} score: {score:.4f}")

    # Test term similarity
    print("\n--- Term Similarity Test ---")
    if "protein" in lsi.term_to_id and "cell" in lsi.term_to_id:
        sim = lsi.get_term_similarity("protein", "cell")
        print(f"Similarity('protein', 'cell'): {sim:.4f}")

    # Find similar terms
    print("\n--- Similar Terms ---")
    test_terms = ["protein", "cell", "blood"]
    for term in test_terms:
        if term in lsi.term_to_id:
            similar = lsi.find_similar_terms(term, k=5)
            print(f"Similar to '{term}':")
            for sim_term, score in similar:
                print(f"  {sim_term}: {score:.4f}")
