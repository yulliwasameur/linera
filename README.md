# LinEra — lineage-complete, forward-secure erasure for LLM-agent memory

Python ≥ 3.11. Seed 20261109 everywhere. Machine used for the paper: 2-core Intel Xeon 2.1 GHz, 7.8 GiB RAM, no GPU.

## Library
- `lf/pprf.py` GGM puncturable PRF (32-bit tags) · `lf/keydag.py` key DAG (iterative derivation)
- `lf/store.py` encrypted SQLite store, atomic state, forget, compaction (`rotate`)
- `lf/vindex.py` encrypted sharded vector index (shard keys follow the lineage)
- `lf/lineage.py` sound / attributed lineage capture · `integrations/mem0_linera.py` mem0 2.2.1 back-end
- `baselines/stores.py` (synthetic pilot) · `agent/memory.py` logical / per-object / epoch / LinEra back-ends
- `tests/` 12 tests: `python3 -m pytest -q tests`

## Reproduce the paper
```
pip install cryptography hnswlib fastembed llama-cpp-python mem0ai numpy pytest
# models/qwen2.5-1.5b-instruct-q4_k_m.gguf  (Qwen/Qwen2.5-1.5B-Instruct-GGUF)
# ../data/locomo10.json (snap-research/locomo), ../data/longmemeval_s_cleaned.json (xiaowu0162/longmemeval-cleaned)
python3 -I agent/build_log.py --convs 0,1,2,3,4          # phase A, LLM writes the memory (~1.5 h CPU)
python3 -I agent/build_window.py --convs 0,1 --w 4       # windowed writer
python3 -I agent/build_lme.py                            # LongMemEval-S subset
./run_queue.sh ; ./run_tail.sh                           # replays, QA, mem0, scaling (all cached/resumable)
python3 -I eval/radius.py
python3 -I eval/make_tables.py --res results --tex ../manuscrit   # every number of the paper
```
`runs/*.jsonl` are the logs of every LLM call (prompt context, raw output, tokens, time);
`results/*.json` hold every number; `results/*_llmcache.json` hold every QA / re-derivation answer.

## Extended campaign (Qwen2.5-7B-Instruct, all ten LoCoMo conversations)
See `gpu/README.md`. Its logs are `runs/*qwen7b*` and its numbers `results/*qwen7b*`.

## Data and licences
The code is released under the MIT licence (`LICENSE`). The logs in `runs/` and the cached answers in `results/`
contain text derived from LoCoMo (snap-research/locomo) and LongMemEval (xiaowu0162/longmemeval) and are released
under the terms of those datasets. The datasets themselves are not redistributed: download them from their sources.

## Citation
See `CITATION.cff`. Paper: Y. Ameur, B. Mokhtari, S. Bouzefrane, "Forget the Lineage: Forward-Secure,
Lineage-Complete Cryptographic Erasure for LLM-Agent Memory", submitted to the Journal of Information Security and
Applications (2026).

## Acknowledgements
This work was supported by a French government grant managed by the Agence Nationale de la Recherche under the
"Investissements d'avenir" program (reference "ANR-21-ESRE-0051"). Centre de Calcul Intensif d'Aix-Marseille is
acknowledged for granting access to its high performance computing resources.
