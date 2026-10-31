# P18 model-card text similarity (child's own text vs candidate card)

Cards kept: 758,866; child documents: 422,527 (median 2 tokens after removing templates and names; 18.6% with >= 5 tokens); name tokens removed: 965; vocabulary 76,784.

chosen_pct_rank: percentile of the chosen parent among the sampled candidates with text (1 = most similar; 0.5 = random; unweighted by log q).

| sample                      |   choices |   share_child_text |   share_rows_with_both_texts |   sim_chosen |   sim_other |   chosen_pct_rank |
|:----------------------------|----------:|-------------------:|-----------------------------:|-------------:|------------:|------------------:|
| all choices                 |     42265 |              0.186 |                        0.121 |        0.050 |       0.014 |             0.554 |
| first author-parent choices |     13360 |              0.259 |                        0.190 |        0.073 |       0.016 |             0.611 |
