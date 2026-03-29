from bsbi import BSBIIndex
from compression import VBEPostings, OptPForDeltaPostings

# Import SPIMI and FST-based indexes (bonus features)
try:
    from spimi import SPIMIIndex
    from fst_index import FSTIndex
    BONUS_AVAILABLE = True
except ImportError:
    BONUS_AVAILABLE = False

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

# SPIMI and FST instances (bonus features)
if BONUS_AVAILABLE:
    SPIMI_instance = SPIMIIndex(data_dir='collection',
                                postings_encoding=VBEPostings,
                                output_dir='index',
                                index_name='spimi_main_index')

    FST_instance = FSTIndex(data_dir='collection',
                            postings_encoding=VBEPostings,
                            output_dir='index',
                            index_name='fst_main_index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

def search_tfidf(queries, k=10):
    """Search menggunakan TF-IDF scoring"""
    print("=" * 60)
    print("TF-IDF RETRIEVAL")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
            print(f"  {doc:30} {score:>.3f}")


def search_bm25(queries, k=10, k1=1.5, b=0.75):
    """Search menggunakan BM25 scoring"""
    print("=" * 60)
    print(f"BM25 RETRIEVAL (k1={k1}, b={b})")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=k1, b=b):
            print(f"  {doc:30} {score:>.3f}")


def search_bm25_wand(queries, k=10, k1=1.5, b=0.75):
    """Search menggunakan BM25 dengan WAND optimization"""
    print("=" * 60)
    print(f"BM25 + WAND RETRIEVAL (k1={k1}, b={b})")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_bm25_wand(query, k=k, k1=k1, b=b):
            print(f"  {doc:30} {score:>.3f}")


def compare_methods(query, k=10):
    """Bandingkan hasil dari ketiga metode retrieval"""
    print("=" * 60)
    print(f"PERBANDINGAN METODE RETRIEVAL")
    print(f"Query: {query}")
    print("=" * 60)

    print("\n--- TF-IDF ---")
    tfidf_results = BSBI_instance.retrieve_tfidf(query, k=k)
    for i, (score, doc) in enumerate(tfidf_results, 1):
        print(f"  {i:2}. {doc:30} {score:>.3f}")

    print("\n--- BM25 ---")
    bm25_results = BSBI_instance.retrieve_bm25(query, k=k)
    for i, (score, doc) in enumerate(bm25_results, 1):
        print(f"  {i:2}. {doc:30} {score:>.3f}")

    print("\n--- BM25 + WAND ---")
    wand_results = BSBI_instance.retrieve_bm25_wand(query, k=k)
    for i, (score, doc) in enumerate(wand_results, 1):
        print(f"  {i:2}. {doc:30} {score:>.3f}")

    # Verifikasi BM25 dan WAND menghasilkan hasil yang sama
    bm25_docs = [doc for _, doc in bm25_results]
    wand_docs = [doc for _, doc in wand_results]

    if bm25_docs == wand_docs:
        print("\n[OK] BM25 dan BM25+WAND menghasilkan ranking yang sama!")
    else:
        print("\n[INFO] BM25 dan BM25+WAND menghasilkan ranking berbeda (normal untuk tie-breaking)")


# ============================================================
# BONUS FEATURES: SPIMI and FST-based search
# ============================================================

def search_spimi(queries, k=10):
    """Search menggunakan SPIMI index (Bonus Feature)"""
    if not BONUS_AVAILABLE:
        print("SPIMI module not available")
        return

    print("=" * 60)
    print("SPIMI INDEX RETRIEVAL (Bonus Feature)")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results (BM25):")
        for (score, doc) in SPIMI_instance.retrieve_bm25(query, k=k):
            print(f"  {doc:30} {score:>.3f}")


def search_fst(queries, k=10):
    """Search menggunakan FST-based index (Bonus Feature)"""
    if not BONUS_AVAILABLE:
        print("FST module not available")
        return

    print("=" * 60)
    print("FST INDEX RETRIEVAL (Bonus Feature)")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results (BM25):")
        for (score, doc) in FST_instance.retrieve_bm25(query, k=k):
            print(f"  {doc:30} {score:>.3f}")


def fst_prefix_search(prefix, k=10):
    """Prefix search using FST (Bonus Feature)"""
    if not BONUS_AVAILABLE:
        print("FST module not available")
        return

    print("=" * 60)
    print(f"FST PREFIX SEARCH: '{prefix}'")
    print("=" * 60)
    results = FST_instance.prefix_search(prefix, k=k)
    for term, term_id in results:
        print(f"  {term} (id={term_id})")


def fst_spell_correct(query, k=10):
    """Spell-corrected search using FST (Bonus Feature)"""
    if not BONUS_AVAILABLE:
        print("FST module not available")
        return

    print("=" * 60)
    print(f"FST SPELL-CORRECTED SEARCH")
    print(f"Original query: '{query}'")
    print("=" * 60)

    results, corrections = FST_instance.spell_corrected_search(query, k=k)

    if corrections:
        print(f"Corrections: {corrections}")
    else:
        print("No corrections needed")

    print("\nResults:")
    for (score, doc) in results:
        print(f"  {doc:30} {score:>.3f}")


if __name__ == "__main__":
    # Demo semua metode
    print("\n" + "=" * 60)
    print("SEARCH ENGINE DEMO")
    print("Fitur: TF-IDF, BM25, BM25+WAND")
    if BONUS_AVAILABLE:
        print("Bonus: SPIMI Index, FST Index (prefix search, spell correction)")
    print("=" * 60 + "\n")

    # Test TF-IDF
    search_tfidf(queries[:1], k=5)
    print()

    # Test BM25
    search_bm25(queries[:1], k=5)
    print()

    # Test BM25 + WAND
    search_bm25_wand(queries[:1], k=5)
    print()

    # Perbandingan
    compare_methods(queries[0], k=5)

    # Bonus features demo
    if BONUS_AVAILABLE:
        print("\n" + "=" * 60)
        print("BONUS FEATURES DEMO")
        print("=" * 60)

        # Note: These require running the bonus indexers first
        # Run: python spimi.py (or python fst_index.py) to create the indexes

        try:
            # FST prefix search
            print("\n--- FST Prefix Search ---")
            fst_prefix_search("meta", k=5)

            # FST spell correction
            print("\n--- FST Spell Correction ---")
            fst_spell_correct("protien metablism", k=3)
        except Exception as e:
            print(f"\nNote: Run 'python fst_index.py' first to create FST index")
            print(f"Error: {e}")
