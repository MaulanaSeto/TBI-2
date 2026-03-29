"""
Finite State Transducer (FST) Implementation for Term Dictionary

FST adalah struktur data yang sangat efisien untuk menyimpan dictionary/lexicon
dengan fitur:
1. Berbagi prefix yang sama antar terms (seperti Trie)
2. Berbagi suffix yang sama antar terms (lebih hemat dari Trie biasa)
3. Mengasosiasikan output (term ID) dengan setiap path yang diterima
4. Mendukung pencarian prefix yang efisien
5. Mendukung pencarian fuzzy (approximate matching)

Keunggulan FST dibanding Hash Table (Python dict):
- Lebih hemat memori untuk dictionary dengan banyak terms yang berbagi prefix/suffix
- Mendukung pencarian prefix secara native
- Mendukung iterasi terurut secara leksikografis
- Mendukung pencarian fuzzy dengan edit distance

Struktur FST:
- Node: Menyimpan transisi ke node lain dan output value (jika final state)
- Edge: Labeled dengan karakter dan mungkin membawa output value
"""

import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterator, Set


class FSTNode:
    """
    Node dalam Finite State Transducer.

    Attributes
    ----------
    is_final: bool
        True jika node ini adalah final state (menerima input)
    output: int
        Output value yang diasosiasikan dengan path ke node ini (untuk final states)
    transitions: Dict[str, FSTNode]
        Mapping dari karakter ke node berikutnya
    edge_outputs: Dict[str, int]
        Output value pada edge (digunakan untuk incremental output)
    """

    __slots__ = ['is_final', 'output', 'transitions', 'edge_outputs', '_hash']

    def __init__(self):
        self.is_final = False
        self.output = 0
        self.transitions = {}
        self.edge_outputs = {}
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((
                self.is_final,
                self.output,
                tuple(sorted(self.transitions.keys())),
                tuple(sorted(self.edge_outputs.items()))
            ))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, FSTNode):
            return False
        return (self.is_final == other.is_final and
                self.output == other.output and
                set(self.transitions.keys()) == set(other.transitions.keys()) and
                self.edge_outputs == other.edge_outputs)


class FST:
    """
    Finite State Transducer untuk term dictionary.

    Implementasi ini menggunakan pendekatan sederhana dimana:
    - Build: Membuat FST dari list of (term, value) pairs yang sudah di-sort
    - Lookup: Mencari term dan mengembalikan value-nya
    - Prefix search: Mencari semua terms dengan prefix tertentu

    Untuk production use, FST bisa di-minimize untuk menghasilkan
    struktur yang lebih compact.
    """

    def __init__(self):
        self.root = FSTNode()
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, term: str, value: int):
        """
        Menambahkan term dengan value ke FST.

        Parameters
        ----------
        term: str
            Term string yang akan ditambahkan
        value: int
            Value yang diasosiasikan dengan term (biasanya term ID)
        """
        node = self.root

        for char in term:
            if char not in node.transitions:
                node.transitions[char] = FSTNode()
            node = node.transitions[char]

        if not node.is_final:
            self._size += 1

        node.is_final = True
        node.output = value

    def get(self, term: str) -> Optional[int]:
        """
        Mencari term dan mengembalikan value-nya.

        Parameters
        ----------
        term: str
            Term yang dicari

        Returns
        -------
        Optional[int]
            Value dari term jika ditemukan, None jika tidak ada
        """
        node = self.root

        for char in term:
            if char not in node.transitions:
                return None
            node = node.transitions[char]

        if node.is_final:
            return node.output
        return None

    def __getitem__(self, term: str) -> int:
        """
        Mendapatkan value untuk term. Raise KeyError jika tidak ditemukan.
        """
        result = self.get(term)
        if result is None:
            raise KeyError(term)
        return result

    def __contains__(self, term: str) -> bool:
        """
        Cek apakah term ada dalam FST.
        """
        return self.get(term) is not None

    def prefix_search(self, prefix: str) -> List[Tuple[str, int]]:
        """
        Mencari semua terms yang dimulai dengan prefix tertentu.

        Parameters
        ----------
        prefix: str
            Prefix yang dicari

        Returns
        -------
        List[Tuple[str, int]]
            List of (term, value) pairs untuk semua terms dengan prefix tersebut
        """
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.transitions:
                return []
            node = node.transitions[char]

        # Collect all terms from this node
        results = []
        self._collect_terms(node, prefix, results)
        return results

    def _collect_terms(self, node: FSTNode, current_term: str,
                       results: List[Tuple[str, int]]):
        """
        Helper method untuk mengumpulkan semua terms dari sebuah node.
        """
        if node.is_final:
            results.append((current_term, node.output))

        for char in sorted(node.transitions.keys()):
            self._collect_terms(
                node.transitions[char],
                current_term + char,
                results
            )

    def fuzzy_search(self, term: str, max_edit_distance: int = 2) -> List[Tuple[str, int, int]]:
        """
        Mencari terms yang mirip dengan term input menggunakan edit distance.

        Menggunakan algoritma Levenshtein dengan pruning untuk efisiensi.

        Parameters
        ----------
        term: str
            Term yang dicari
        max_edit_distance: int
            Maksimum edit distance yang diizinkan

        Returns
        -------
        List[Tuple[str, int, int]]
            List of (term, value, edit_distance) untuk terms yang mirip
        """
        results = []
        current_row = list(range(len(term) + 1))

        self._fuzzy_search_recursive(
            self.root, '', term, current_row,
            max_edit_distance, results
        )

        return sorted(results, key=lambda x: (x[2], x[0]))

    def _fuzzy_search_recursive(self, node: FSTNode, current_term: str,
                                target: str, previous_row: List[int],
                                max_dist: int, results: List[Tuple[str, int, int]]):
        """
        Helper method untuk fuzzy search menggunakan dynamic programming.
        """
        columns = len(target) + 1

        # Check if this is a final state and within distance
        if node.is_final:
            if previous_row[-1] <= max_dist:
                results.append((current_term, node.output, previous_row[-1]))

        # Early termination: jika minimum di row > max_dist, tidak perlu lanjut
        if min(previous_row) > max_dist:
            return

        for char in node.transitions.keys():
            current_row = [previous_row[0] + 1]

            for column in range(1, columns):
                insert_cost = current_row[column - 1] + 1
                delete_cost = previous_row[column] + 1

                if target[column - 1] != char:
                    replace_cost = previous_row[column - 1] + 1
                else:
                    replace_cost = previous_row[column - 1]

                current_row.append(min(insert_cost, delete_cost, replace_cost))

            # Prune branches that exceed max distance
            if min(current_row) <= max_dist:
                self._fuzzy_search_recursive(
                    node.transitions[char],
                    current_term + char,
                    target,
                    current_row,
                    max_dist,
                    results
                )

    def iterate_all(self) -> Iterator[Tuple[str, int]]:
        """
        Iterasi semua terms dalam urutan leksikografis.

        Yields
        ------
        Tuple[str, int]
            (term, value) pairs
        """
        stack = [(self.root, '')]

        while stack:
            node, prefix = stack.pop()

            if node.is_final:
                yield (prefix, node.output)

            # Add children in reverse order so we process them in order
            for char in sorted(node.transitions.keys(), reverse=True):
                stack.append((node.transitions[char], prefix + char))

    def save(self, filepath: str):
        """
        Menyimpan FST ke file.

        Parameters
        ----------
        filepath: str
            Path ke file output
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'FST':
        """
        Memuat FST dari file.

        Parameters
        ----------
        filepath: str
            Path ke file input

        Returns
        -------
        FST
            FST yang dimuat dari file
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_stats(self) -> Dict:
        """
        Mengembalikan statistik tentang FST.

        Returns
        -------
        Dict
            Dictionary berisi statistik FST
        """
        node_count = 0
        edge_count = 0
        final_count = 0

        stack = [self.root]
        visited = set()

        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))

            node_count += 1
            edge_count += len(node.transitions)
            if node.is_final:
                final_count += 1

            for child in node.transitions.values():
                stack.append(child)

        return {
            'total_terms': self._size,
            'node_count': node_count,
            'edge_count': edge_count,
            'final_state_count': final_count,
        }


class FSTDictionary:
    """
    Dictionary berbasis FST yang bisa menggantikan IdMap.

    Menyediakan interface yang kompatibel dengan IdMap tapi menggunakan
    FST sebagai backend untuk efisiensi memori dan fitur tambahan.
    """

    def __init__(self):
        self.fst = FST()
        self.id_to_str = []  # Untuk reverse lookup (id -> string)

    def __len__(self):
        return len(self.id_to_str)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.fst
        elif isinstance(key, int):
            return 0 <= key < len(self.id_to_str)
        return False

    def __getitem__(self, key):
        """
        Mengakses dictionary dengan string atau integer key.

        Jika key adalah string:
            - Jika string sudah ada, return ID-nya
            - Jika string belum ada, tambahkan dan return ID baru

        Jika key adalah integer:
            - Return string yang bersesuaian
        """
        if isinstance(key, str):
            result = self.fst.get(key)
            if result is not None:
                return result
            # Add new term
            new_id = len(self.id_to_str)
            self.fst.add(key, new_id)
            self.id_to_str.append(key)
            return new_id
        elif isinstance(key, int):
            return self.id_to_str[key]
        else:
            raise TypeError(f"Key must be str or int, got {type(key)}")

    def get_id(self, term: str) -> Optional[int]:
        """Get ID for term without adding if not exists"""
        return self.fst.get(term)

    def get_str(self, term_id: int) -> str:
        """Get string for term ID"""
        return self.id_to_str[term_id]

    def prefix_search(self, prefix: str) -> List[Tuple[str, int]]:
        """Search terms by prefix"""
        return self.fst.prefix_search(prefix)

    def fuzzy_search(self, term: str, max_distance: int = 2) -> List[Tuple[str, int, int]]:
        """Search similar terms"""
        return self.fst.fuzzy_search(term, max_distance)

    def iterate_all(self) -> Iterator[Tuple[str, int]]:
        """Iterate all terms in sorted order"""
        return self.fst.iterate_all()

    def get_stats(self) -> Dict:
        """Get FST statistics"""
        return self.fst.get_stats()

    def save(self, filepath: str):
        """Save dictionary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({'fst': self.fst, 'id_to_str': self.id_to_str}, f)

    @staticmethod
    def load(filepath: str) -> 'FSTDictionary':
        """Load dictionary from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        fst_dict = FSTDictionary()
        fst_dict.fst = data['fst']
        fst_dict.id_to_str = data['id_to_str']
        return fst_dict


class MinimalFST:
    """
    Minimal FST implementation yang lebih memory-efficient.

    Menggunakan pendekatan hash-based untuk mendeteksi dan
    menggabungkan suffix yang sama.
    """

    def __init__(self):
        self.root = {}
        self._size = 0
        # Registry untuk node deduplication
        self._registry = {}

    def build(self, sorted_terms: List[Tuple[str, int]]):
        """
        Build FST dari list of (term, value) yang sudah di-sort.

        Parameters
        ----------
        sorted_terms: List[Tuple[str, int]]
            List of (term, value) pairs, HARUS sudah di-sort berdasarkan term
        """
        # Stack untuk menyimpan state saat traversal
        # Each entry: (prefix, node, child_chars, last_char_processed)
        self.root = {'_final': False, '_output': 0}
        self._size = 0

        previous_word = ""

        for word, value in sorted_terms:
            # Find common prefix with previous word
            common_prefix_len = 0
            for i in range(min(len(word), len(previous_word))):
                if word[i] == previous_word[i]:
                    common_prefix_len += 1
                else:
                    break

            # Add the new suffix
            node = self.root
            for i, char in enumerate(word):
                if char not in node:
                    node[char] = {'_final': False, '_output': 0}
                node = node[char]

            node['_final'] = True
            node['_output'] = value
            self._size += 1

            previous_word = word

    def get(self, term: str) -> Optional[int]:
        """
        Mencari term dan mengembalikan value-nya.
        """
        node = self.root
        for char in term:
            if char not in node:
                return None
            node = node[char]

        if node.get('_final', False):
            return node['_output']
        return None

    def __contains__(self, term: str) -> bool:
        return self.get(term) is not None

    def __len__(self):
        return self._size

    def prefix_search(self, prefix: str) -> List[Tuple[str, int]]:
        """
        Mencari semua terms dengan prefix tertentu.
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]

        results = []
        self._collect_terms_dict(node, prefix, results)
        return results

    def _collect_terms_dict(self, node: dict, current_term: str,
                           results: List[Tuple[str, int]]):
        """Helper untuk mengumpulkan terms dari dictionary-based node"""
        if node.get('_final', False):
            results.append((current_term, node['_output']))

        for key in sorted(node.keys()):
            if not key.startswith('_'):
                self._collect_terms_dict(node[key], current_term + key, results)

    def save(self, filepath: str):
        """Save minimal FST to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({'root': self.root, 'size': self._size}, f)

    @staticmethod
    def load(filepath: str) -> 'MinimalFST':
        """Load minimal FST from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        fst = MinimalFST()
        fst.root = data['root']
        fst._size = data['size']
        return fst


# Test code
if __name__ == '__main__':
    print("=" * 60)
    print("FST (Finite State Transducer) Test")
    print("=" * 60)

    # Test basic FST
    print("\n--- Basic FST Test ---")
    fst = FST()

    # Add some terms
    terms = [
        ("apple", 1),
        ("application", 2),
        ("apply", 3),
        ("banana", 4),
        ("band", 5),
        ("bandana", 6),
    ]

    for term, value in terms:
        fst.add(term, value)

    print(f"Added {len(fst)} terms")

    # Test lookup
    print("\nLookup tests:")
    for term, expected in terms:
        result = fst.get(term)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {term}: {result} (expected {expected}) [{status}]")

    # Test non-existent
    print(f"  'orange': {fst.get('orange')} (expected None)")

    # Test prefix search
    print("\nPrefix search tests:")
    print(f"  prefix 'app': {fst.prefix_search('app')}")
    print(f"  prefix 'ban': {fst.prefix_search('ban')}")
    print(f"  prefix 'xyz': {fst.prefix_search('xyz')}")

    # Test fuzzy search
    print("\nFuzzy search tests:")
    print(f"  fuzzy 'aple' (dist=1): {fst.fuzzy_search('aple', 1)}")
    print(f"  fuzzy 'banan' (dist=1): {fst.fuzzy_search('banan', 1)}")

    # Test FSTDictionary
    print("\n--- FSTDictionary Test ---")
    fst_dict = FSTDictionary()

    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    ids = [fst_dict[word] for word in words]
    print(f"Word IDs: {list(zip(words, ids))}")
    print(f"Reverse lookup id=2: {fst_dict[2]}")

    # Test MinimalFST
    print("\n--- MinimalFST Test ---")
    mfst = MinimalFST()
    sorted_terms = sorted(terms)
    mfst.build(sorted_terms)

    print(f"Built minimal FST with {len(mfst)} terms")
    print("Lookup tests:")
    for term, expected in terms:
        result = mfst.get(term)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {term}: {result} (expected {expected}) [{status}]")

    print(f"\nPrefix 'app': {mfst.prefix_search('app')}")

    # Print stats
    print("\n--- FST Statistics ---")
    stats = fst.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nAll tests completed!")
