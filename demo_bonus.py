"""
Bonus Features Demo Script

This script demonstrates the bonus features implemented for TP2:
1. SPIMI (Single-Pass In-Memory Indexing)
2. FST (Finite State Transducer) for dictionary
3. LSI (Latent Semantic Indexing) with Vector Indexing

Run this script to see all features in action.
"""

import os
import time
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression import VBEPostings, OptPForDeltaPostings


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n--- {title} ---")


def demo_spimi():
    """Demonstrate SPIMI indexing"""
    from spimi import SPIMIIndex

    print_header("SPIMI (Single-Pass In-Memory Indexing) Demo")

    print("""
SPIMI vs BSBI Differences:

+----------------------+----------------------------------+--------------------------------+
| Aspect               | BSBI                             | SPIMI                          |
+----------------------+----------------------------------+--------------------------------+
| Term representation  | Uses global term_id_map          | Uses term strings directly     |
|                      | during indexing                  | during indexing                |
+----------------------+----------------------------------+--------------------------------+
| Memory usage         | Needs term_id_map to fit         | Dictionary built on-the-fly   |
|                      | in memory                        | per block                      |
+----------------------+----------------------------------+--------------------------------+
| Index construction   | Sort (term_id, doc_id) pairs     | Add to dictionary directly     |
+----------------------+----------------------------------+--------------------------------+
| Merge                | Merge by term_id                 | Merge by term string           |
+----------------------+----------------------------------+--------------------------------+
| Global term_id       | Assigned during parsing          | Assigned during final merge    |
+----------------------+----------------------------------+--------------------------------+
""")

    print("Initializing SPIMI Index...")
    spimi = SPIMIIndex(
        data_dir='collection',
        postings_encoding=VBEPostings,
        output_dir='index',
        index_name='spimi_main_index',
        memory_threshold=30000  # Smaller threshold for demo
    )

    print("\nRunning SPIMI indexing...")
    start_time = time.time()
    spimi.index()
    elapsed = time.time() - start_time
    print(f"SPIMI indexing completed in {elapsed:.2f} seconds")

    # Test retrieval
    print_subheader("SPIMI Retrieval Test")

    queries = [
        "alkylated with radioactive iodoacetate",
        "lipid metabolism in toxemia",
        "psychodrama for disturbed children"
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        # Load index if needed
        if len(spimi.term_id_map) == 0:
            spimi.load()

        print("  TF-IDF Results:")
        tfidf_results = spimi.retrieve_tfidf(query, k=3)
        for score, doc in tfidf_results:
            print(f"    {os.path.basename(doc):30} score: {score:.4f}")

        print("  BM25 Results:")
        bm25_results = spimi.retrieve_bm25(query, k=3)
        for score, doc in bm25_results:
            print(f"    {os.path.basename(doc):30} score: {score:.4f}")


def demo_fst():
    """Demonstrate Finite State Transducer"""
    from fst import FST, FSTDictionary

    print_header("FST (Finite State Transducer) Demo")

    print("""
FST Advantages over Hash Table:

+----------------------+----------------------------------+--------------------------------+
| Feature              | Hash Table (dict)                | FST                            |
+----------------------+----------------------------------+--------------------------------+
| Memory               | O(n * avg_term_length)           | Shared prefixes/suffixes       |
+----------------------+----------------------------------+--------------------------------+
| Prefix search        | O(n) - scan all keys             | O(prefix_len + results)        |
+----------------------+----------------------------------+--------------------------------+
| Fuzzy search         | O(n * term_len) - check all      | Efficient with pruning         |
+----------------------+----------------------------------+--------------------------------+
| Ordered iteration    | Need to sort keys                | Built-in ordered traversal     |
+----------------------+----------------------------------+--------------------------------+
| Autocomplete         | Not supported                    | Fast prefix enumeration        |
+----------------------+----------------------------------+--------------------------------+
""")

    # Build FST with sample data
    print_subheader("Building FST")

    fst = FST()
    terms = [
        ("algorithm", 1),
        ("alphabetical", 2),
        ("alpha", 3),
        ("apartment", 4),
        ("application", 5),
        ("apply", 6),
        ("approximate", 7),
        ("banana", 8),
        ("band", 9),
        ("bandana", 10),
        ("biology", 11),
        ("bioinformatics", 12),
        ("computer", 13),
        ("computation", 14),
        ("compute", 15),
    ]

    for term, value in terms:
        fst.add(term, value)

    print(f"Added {len(fst)} terms to FST")
    stats = fst.get_stats()
    print(f"FST Statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Final states: {stats['final_state_count']}")

    # Demonstrate prefix search
    print_subheader("Prefix Search")

    prefixes = ["alp", "app", "com", "bio"]
    for prefix in prefixes:
        results = fst.prefix_search(prefix)
        print(f"  Prefix '{prefix}': {[t for t, _ in results]}")

    # Demonstrate fuzzy search
    print_subheader("Fuzzy Search (Spelling Correction)")

    typos = [
        ("algoritm", 1),   # typo for "algorithm"
        ("computr", 1),    # typo for "computer"
        ("aplicaton", 2),  # typo for "application"
    ]

    for typo, max_dist in typos:
        results = fst.fuzzy_search(typo, max_dist)
        print(f"  '{typo}' (max_dist={max_dist}):")
        for term, tid, dist in results:
            print(f"    -> '{term}' (distance={dist})")


def demo_fst_index():
    """Demonstrate FST-based indexing with search features"""
    from fst_index import FSTIndex

    print_header("FST Index Demo (Advanced Search Features)")

    print("""
FST Index provides additional search capabilities:

1. Prefix Search     - Find all terms starting with a prefix
2. Fuzzy Search      - Find terms similar to query (spelling correction)
3. Query Expansion   - Expand query with related terms
4. Spell Correction  - Automatic typo correction during search
""")

    print("Initializing FST Index...")
    fst_index = FSTIndex(
        data_dir='collection',
        postings_encoding=VBEPostings,
        output_dir='index',
        index_name='fst_main_index'
    )

    print("\nRunning FST-based indexing...")
    start_time = time.time()
    fst_index.index()
    elapsed = time.time() - start_time
    print(f"FST indexing completed in {elapsed:.2f} seconds")

    # Standard retrieval
    print_subheader("Standard BM25 Retrieval")

    query = "lipid metabolism"
    print(f"Query: {query}")
    results = fst_index.retrieve_bm25(query, k=5)
    for score, doc in results:
        print(f"  {os.path.basename(doc):30} score: {score:.4f}")

    # Prefix search
    print_subheader("Prefix Search Feature")

    prefixes = ["meta", "prot", "cell"]
    for prefix in prefixes:
        results = fst_index.prefix_search(prefix, k=5)
        print(f"  Prefix '{prefix}':")
        terms = [term for term, _ in results[:5]]
        print(f"    {terms}")

    # Fuzzy term search
    print_subheader("Fuzzy Term Search (Vocabulary Lookup)")

    typos = ["protien", "metablism", "ceel"]
    for typo in typos:
        results = fst_index.fuzzy_term_search(typo, max_distance=2)
        if results:
            print(f"  '{typo}' -> ")
            for term, tid, dist in results[:3]:
                print(f"    '{term}' (edit_distance={dist})")
        else:
            print(f"  '{typo}' -> No matches found")

    # Spell-corrected search
    print_subheader("Spell-Corrected Search")

    queries_with_typos = [
        "protien metablism",
        "ceel divison",
    ]

    for query in queries_with_typos:
        print(f"\n  Original query: '{query}'")
        results, corrections = fst_index.spell_corrected_search(query, k=3)

        if corrections:
            print(f"  Corrections: {corrections}")
        else:
            print(f"  No corrections needed")

        print(f"  Results:")
        for score, doc in results:
            print(f"    {os.path.basename(doc):30} score: {score:.4f}")


def demo_comparison():
    """Compare BSBI, SPIMI, and FST-based indexing"""
    from bsbi import BSBIIndex
    from spimi import SPIMIIndex
    from fst_index import FSTIndex

    print_header("Indexing Methods Comparison")

    methods = [
        ("BSBI", BSBIIndex, {'data_dir': 'collection', 'postings_encoding': VBEPostings,
                            'output_dir': 'index', 'index_name': 'compare_bsbi'}),
        ("SPIMI", SPIMIIndex, {'data_dir': 'collection', 'postings_encoding': VBEPostings,
                              'output_dir': 'index', 'index_name': 'compare_spimi',
                              'memory_threshold': 50000}),
        ("FST Index", FSTIndex, {'data_dir': 'collection', 'postings_encoding': VBEPostings,
                                 'output_dir': 'index', 'index_name': 'compare_fst'}),
    ]

    query = "alkylated with radioactive iodoacetate"
    print(f"\nTest query: {query}\n")

    for name, IndexClass, params in methods:
        print(f"{name}:")
        idx = IndexClass(**params)

        # Index
        start = time.time()
        idx.index()
        index_time = time.time() - start

        # Search
        if len(idx.term_id_map) == 0 if hasattr(idx, 'term_id_map') else len(idx.term_fst) == 0:
            idx.load()

        start = time.time()
        results = idx.retrieve_bm25(query, k=3)
        search_time = time.time() - start

        print(f"  Index time: {index_time:.2f}s")
        print(f"  Search time: {search_time:.4f}s")
        print(f"  Top result: {os.path.basename(results[0][1]) if results else 'None'}")
        print()


def demo_lsi():
    """Demonstrate Latent Semantic Indexing"""
    from lsi import LSIIndex

    print_header("LSI (Latent Semantic Indexing) Demo")

    print("""
LSI (Latent Semantic Indexing) uses SVD to find latent semantic structure:

+----------------------+----------------------------------+--------------------------------+
| Feature              | Traditional (TF-IDF/BM25)        | LSI                            |
+----------------------+----------------------------------+--------------------------------+
| Representation       | Sparse term vectors              | Dense semantic vectors         |
+----------------------+----------------------------------+--------------------------------+
| Synonymy             | Cannot handle                    | Similar terms cluster together |
+----------------------+----------------------------------+--------------------------------+
| Polysemy             | Cannot distinguish               | Context-dependent meaning      |
+----------------------+----------------------------------+--------------------------------+
| Storage              | O(vocab_size)                    | O(n_components)                |
+----------------------+----------------------------------+--------------------------------+
| Similarity           | Exact term matching              | Semantic similarity            |
+----------------------+----------------------------------+--------------------------------+

Process:
1. Build Term-Document Matrix (TDM) with TF-IDF weighting
2. Apply Truncated SVD: A ≈ U_k × Σ_k × V_k^T
3. Document vectors = V_k × Σ_k
4. Query projection: q' = q^T × U_k × Σ_k^(-1)
5. Similarity search using cosine similarity
""")

    print("Initializing LSI Index...")
    lsi = LSIIndex(
        data_dir='collection',
        output_dir='index',
        n_components=50  # Smaller for demo
    )

    print("\nBuilding LSI index...")
    start_time = time.time()
    lsi.index()
    elapsed = time.time() - start_time
    print(f"LSI indexing completed in {elapsed:.2f} seconds")

    # Test retrieval
    print_subheader("LSI Retrieval Test")

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

    # Term similarity
    print_subheader("Semantic Term Similarity")

    term_pairs = [
        ("protein", "cell"),
        ("blood", "plasma"),
        ("disease", "treatment"),
    ]

    for term1, term2 in term_pairs:
        if term1 in lsi.term_to_id and term2 in lsi.term_to_id:
            sim = lsi.get_term_similarity(term1, term2)
            print(f"  similarity('{term1}', '{term2}'): {sim:.4f}")

    # Find similar terms
    print_subheader("Find Similar Terms")

    test_terms = ["protein", "blood", "cell"]
    for term in test_terms:
        if term in lsi.term_to_id:
            similar = lsi.find_similar_terms(term, k=5)
            print(f"\n  Similar to '{term}':")
            for sim_term, score in similar:
                print(f"    {sim_term:15} {score:.4f}")


def main():
    """Main demo function"""
    print_header("BONUS FEATURES DEMONSTRATION")
    print("""
    This script demonstrates the bonus features implemented for TP2:

    1. SPIMI (Single-Pass In-Memory Indexing)
       - Alternative to BSBI for index construction
       - Uses term strings directly instead of term IDs during indexing
       - More memory-efficient for large vocabularies

    2. FST (Finite State Transducer) for Dictionary
       - Memory-efficient storage of term dictionary
       - Supports prefix search (autocomplete)
       - Supports fuzzy search (spelling correction)
       - Ordered iteration of terms

    3. LSI (Latent Semantic Indexing) with Vector Indexing
       - Uses SVD for dimensionality reduction
       - Finds latent semantic structure
       - Supports semantic similarity search
       - Handles synonymy and polysemy

    Select demo to run:
    1. SPIMI Demo
    2. FST Demo
    3. FST Index Demo (Advanced Search Features)
    4. LSI Demo (Latent Semantic Indexing)
    5. Comparison of All Methods
    6. Run All Demos
    """)

    choice = input("Enter choice (1-6) [6]: ").strip() or "6"

    if choice == "1":
        demo_spimi()
    elif choice == "2":
        demo_fst()
    elif choice == "3":
        demo_fst_index()
    elif choice == "4":
        demo_lsi()
    elif choice == "5":
        demo_comparison()
    elif choice == "6":
        demo_fst()  # FST first (no indexing needed)
        demo_spimi()
        demo_fst_index()
        demo_lsi()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
