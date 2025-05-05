# Balanced Re-Ranker

A Python library for re-ranking ranked lists to optimize relevance while improving fairness across predefined groups or categories. It aims to maximize nDCG@k and minimize AWRF (Attention Weighted Rank Fairness).

## Features

- Fairness-aware re-ranking using Bayesian optimization
- Measures: nDCG and AWRF
- Supports custom groups (e.g., category, source)
- Configurable via lambda, slack, and time limits

## Usage

```python
import random, math
from fair_reranker import FairReranker

random.seed(20250423)
categories = ['CatA', 'CatB', 'CatC', 'CatD']
sources = ['S1', 'S2', 'S3', 'S4']

# Generate 100 synthetic items
data100 = [
    (i, "demo_query", f"doc_text_{i}", random.choice(categories), round(random.random(), 3), random.choice(sources))
    for i in range(1, 101)
]
data100 = sorted(data100, key=lambda x: x[4], reverse=True)
k = 10
orig_topk = data100[:k]
true_ideal = [item[4] for item in data100[:k]]

# Run Fair Re-Ranker
reranker = BalancedReranker(
    k=k, lambda_bounds=(0.1, 3.0), time_limit=20,
    bayes_calls=7, initial_points=7,
    slack_weight=0.2, imbalance_correction=True, verbose=True
)
reranked = reranker.rerank(data100)

# Evaluation
def dcg(rels): return sum(r / math.log2(i+2) for i, r in enumerate(rels))
ndcg_before = dcg([x[4] for x in orig_topk]) / dcg(true_ideal)
ndcg_after = dcg([x[4] for x in reranked]) / dcg(true_ideal)
aw_before = reranker._awrf(data100, orig_topk)
aw_after = reranker._awrf(data100, reranked)

print(f"nDCG@{k}: before={ndcg_before:.4f}, after={ndcg_after:.4f}")
print(f"AWRF    : before={aw_before:.4f}, after={aw_after:.4f}")
