import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> IR metric: DCG (Discounted Cumulative Gain)

def dcg(ranking, k=None):
    """
    Menghitung Discounted Cumulative Gain (DCG).

    DCG mengukur kualitas ranking dengan memberikan bobot lebih tinggi
    pada dokumen relevan yang muncul di posisi atas.

    Formula: DCG@k = Σ (rel_i / log2(i + 1)) untuk i = 1 sampai k

    Dimana:
    - rel_i = relevansi dokumen di posisi i (0 atau 1 untuk binary relevance)
    - log2(i + 1) = discount factor berdasarkan posisi

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, 1, 0]
        1 = relevan, 0 = tidak relevan
    k: int or None
        Cutoff untuk evaluasi. Jika None, gunakan semua dokumen.

    Returns
    -------
    float
        Skor DCG
    """
    if not ranking:
        return 0.0

    if k is None:
        k = len(ranking)
    else:
        k = min(k, len(ranking))

    score = 0.0
    for i in range(k):
        # Posisi dimulai dari 1 (bukan 0)
        # Discount: log2(position + 1) = log2(i + 2) karena i dimulai dari 0
        if ranking[i] == 1:
            score += 1.0 / math.log2(i + 2)

    return score


######## >>>>> IR metric: IDCG (Ideal DCG)

def idcg(ranking, k=None):
    """
    Menghitung Ideal DCG (IDCG).

    IDCG adalah DCG maksimum yang mungkin dicapai jika semua dokumen
    relevan berada di posisi teratas.

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi
    k: int or None
        Cutoff untuk evaluasi

    Returns
    -------
    float
        Skor IDCG
    """
    if not ranking:
        return 0.0

    # Hitung jumlah dokumen relevan
    num_relevant = sum(ranking)

    if num_relevant == 0:
        return 0.0

    if k is None:
        k = len(ranking)

    # Ideal ranking: semua dokumen relevan di atas
    ideal_ranking = [1] * min(num_relevant, k) + [0] * max(0, k - num_relevant)

    return dcg(ideal_ranking, k)


######## >>>>> IR metric: NDCG (Normalized DCG)

def ndcg(ranking, k=None):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG).

    NDCG = DCG / IDCG

    NDCG menghasilkan nilai antara 0 dan 1, dimana:
    - 1 = ranking sempurna (semua dokumen relevan di atas)
    - 0 = tidak ada dokumen relevan

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, 1, 0]
    k: int or None
        Cutoff untuk evaluasi. Jika None, gunakan semua dokumen.

    Returns
    -------
    float
        Skor NDCG (antara 0 dan 1)
    """
    if not ranking:
        return 0.0

    dcg_score = dcg(ranking, k)
    idcg_score = idcg(ranking, k)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score


######## >>>>> IR metric: AP (Average Precision)

def ap(ranking):
    """
    Menghitung Average Precision (AP).

    AP adalah rata-rata dari precision pada setiap posisi dimana
    dokumen relevan ditemukan.

    Formula: AP = (1/R) × Σ P(k) × rel(k)

    Dimana:
    - R = jumlah total dokumen relevan
    - P(k) = precision pada posisi k
    - rel(k) = 1 jika dokumen di posisi k relevan, 0 jika tidak

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, 1, 0]

    Returns
    -------
    float
        Skor Average Precision (antara 0 dan 1)
    """
    if not ranking:
        return 0.0

    num_relevant = sum(ranking)

    if num_relevant == 0:
        return 0.0

    score = 0.0
    num_relevant_found = 0

    for i in range(len(ranking)):
        if ranking[i] == 1:
            num_relevant_found += 1
            # Precision at position i+1 (posisi dimulai dari 1)
            precision_at_k = num_relevant_found / (i + 1)
            score += precision_at_k

    return score / num_relevant


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000, method="tfidf"):
  """
    Loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    Untuk setiap query, kembalikan top-k documents.

    Menghitung 4 metrik evaluasi:
    - RBP (Rank Biased Precision)
    - DCG (Discounted Cumulative Gain)
    - NDCG (Normalized DCG)
    - AP (Average Precision)

    Parameters
    ----------
    qrels: dict
        Query relevance judgments
    query_file: str
        Path ke file queries
    k: int
        Jumlah dokumen yang di-retrieve per query
    method: str
        Metode retrieval: "tfidf" atau "bm25"

    Returns
    -------
    dict
        Dictionary berisi rata-rata skor untuk setiap metrik
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []

      # Pilih metode retrieval
      if method == "bm25":
        results = BSBI_instance.retrieve_bm25(query, k=k)
      else:
        results = BSBI_instance.retrieve_tfidf(query, k=k)

      for (score, doc) in results:
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])

      # Hitung semua metrik
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking, k=10))  # DCG@10
      ndcg_scores.append(ndcg(ranking, k=10))  # NDCG@10
      ap_scores.append(ap(ranking))

  # Hitung rata-rata
  results = {
      "RBP": sum(rbp_scores) / len(rbp_scores),
      "DCG@10": sum(dcg_scores) / len(dcg_scores),
      "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
      "AP": sum(ap_scores) / len(ap_scores)
  }

  print(f"Hasil evaluasi {method.upper()} terhadap {len(rbp_scores)} queries")
  print(f"RBP score     = {results['RBP']:.4f}")
  print(f"DCG@10 score  = {results['DCG@10']:.4f}")
  print(f"NDCG@10 score = {results['NDCG@10']:.4f}")
  print(f"AP score      = {results['AP']:.4f}")

  return results


def eval_all(qrels, query_file="queries.txt", k=1000):
    """
    Evaluasi kedua metode retrieval (TF-IDF dan BM25) sekaligus
    dan bandingkan hasilnya.

    Parameters
    ----------
    qrels: dict
        Query relevance judgments
    query_file: str
        Path ke file queries
    k: int
        Jumlah dokumen yang di-retrieve per query

    Returns
    -------
    dict
        Dictionary berisi hasil evaluasi untuk kedua metode
    """
    print("=" * 50)
    print("EVALUASI TF-IDF vs BM25")
    print("=" * 50)

    print("\n--- TF-IDF ---")
    tfidf_results = eval(qrels, query_file, k, method="tfidf")

    print("\n--- BM25 ---")
    bm25_results = eval(qrels, query_file, k, method="bm25")

    print("\n" + "=" * 50)
    print("PERBANDINGAN")
    print("=" * 50)
    print(f"{'Metric':<12} {'TF-IDF':<12} {'BM25':<12} {'Diff':<12}")
    print("-" * 50)

    for metric in ["RBP", "DCG@10", "NDCG@10", "AP"]:
        tfidf_val = tfidf_results[metric]
        bm25_val = bm25_results[metric]
        diff = bm25_val - tfidf_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{metric:<12} {tfidf_val:<12.4f} {bm25_val:<12.4f} {diff_str:<12}")

    return {"tfidf": tfidf_results, "bm25": bm25_results}

if __name__ == '__main__':
  # Test metrics dengan contoh sederhana
  print("=== Test Metrik Evaluasi ===\n")

  # Contoh ranking: dokumen di posisi 1,3,4,5 relevan
  test_ranking = [1, 0, 1, 1, 1, 0, 0, 1, 0, 0]
  print(f"Test ranking: {test_ranking}")
  print(f"Jumlah relevan: {sum(test_ranking)}")
  print(f"RBP score   : {rbp(test_ranking):.4f}")
  print(f"DCG@10 score: {dcg(test_ranking, k=10):.4f}")
  print(f"NDCG@10 score: {ndcg(test_ranking, k=10):.4f}")
  print(f"AP score    : {ap(test_ranking):.4f}")

  # Test dengan perfect ranking
  perfect_ranking = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  print(f"\nPerfect ranking: {perfect_ranking}")
  print(f"NDCG@10 score: {ndcg(perfect_ranking, k=10):.4f}")  # Harus = 1.0
  print(f"AP score    : {ap(perfect_ranking):.4f}")  # Harus = 1.0

  # Test dengan worst ranking
  worst_ranking = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  print(f"\nWorst ranking: {worst_ranking}")
  print(f"NDCG@10 score: {ndcg(worst_ranking, k=10):.4f}")
  print(f"AP score    : {ap(worst_ranking):.4f}")

  print("\n" + "=" * 50)
  print("=== Evaluasi pada Dataset ===")
  print("=" * 50 + "\n")

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # Evaluasi kedua metode
  eval_all(qrels)