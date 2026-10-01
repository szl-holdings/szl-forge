# Interpretation of the September 12 experiments

The archived experiments are local research observations. The eight templates share two visibly artificial identifiers. V3 recognizes that identifier shape and rejects absent tokens, so zero errors on this family are expected by construction. Changing the seed does not create a new distribution. Corpus absence establishes no exact normalized mention; it does not exclude aliases, paraphrases, or implicit answers in natural language. These results support a narrow input-policy regression check, not general semantic safety or frontier model superiority.

The unchanged V3 replication recorded 3 base answers and 0 guarded answers among 240 synthetic unsupported holdout queries, with no additional abstentions on 48 public positive replays. The nominal Wilson upper bound was 1.58%. Shared templates, deterministic decisions, and correlated anchors mean that number must not be presented as a guarantee for real user traffic. Public positive replay may be contaminated and is not a hidden test.

V2's larger global margin threshold reduced one holdout from 3/80 false answers to 0/80, but positive replay coverage fell from 75% to 20.83%. That policy is unsuitable as the application's default. The integrated identifier filter remains optional; ordinary retrieval and the original calibrated threshold retain their behavior.

No training has occurred. Model files remain upstream Qwen3-Embedding-0.6B and deepset/roberta-base-squad2 at their pinned revisions. The next justified model experiment is a small reranker or learned abstention model trained on rights-reviewed, corpus-disjoint examples. Its admission test must include legitimate unseen identifiers, ordinary absent facts, aliases, Unicode variants, contradictory passages and human relevance judgments, with coverage measured alongside error.

A local SHA-256 receipt checks consistency. It is not an independent signature, witness, or production-provider attestation.
