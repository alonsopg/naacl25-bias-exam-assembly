# Balanced Re-Ranker

Our core method for re-ranking ranked lists to optimize relevance while improving fairness across predefined groups or categories. It aims to maximize nDCG@k and minimize AWRF (Attention Weighted Rank Fairness).

## Features

- Fairness-aware re-ranking using Bayesian optimization
- Measures: nDCG and AWRF
- Supports custom groups (e.g., category, source)
- Configurable via lambda, slack, and time limits

## Usage

```python
import random, math
from reranker import BalancedReranker

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


```shell
INFO - Initialized: k=10, λ_bounds=(0.1, 3.0), time_limit=20s, bayes_calls=7, init_points=7, slack_weight=0.200, imbalance_correction=True
INFO - Starting Bayesian opt: calls=7, init=7
INFO - Bayes 1/7: λ=0.5814 → combined=0.7600
...
INFO - Bayes 7/7: λ=0.3448 → combined=0.8600
INFO - Bayes done: λ∈[0.3448,0.8310] → best=0.3448
INFO - Orig IDs: [66, 4, 22, 59, 67, 17, 76, 84, 72, 33]
INFO - New  IDs: [28, 30, 64, 51, 99, 83, 50, 62, 74, 15]

Original top-10:
 1. id=66, cat=CatC, rel=0.997, src=S3
 2. id=4,  cat=CatC, rel=0.975, src=S3
 ...
10. id=33, cat=CatA, rel=0.916, src=S4

Re-ranked top-10:
 1. id=28, cat=CatD, rel=0.898, src=S3, att=0.256
 2. id=30, cat=CatA, rel=0.823, src=S2, att=0.221
 ...
10. id=15, cat=CatA, rel=0.697, src=S4, att=0.187

true_ideal rels: ['0.997', '0.975', ..., '0.916']
orig_rels   : ['0.997', '0.975', ..., '0.916']
new_rels    : ['0.898', '0.823', ..., '0.697']

nDCG@10: before=1.0000, after=0.8163  
AWRF    : before=0.2500, after=0.1400
```
