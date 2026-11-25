# Dataset audit and provenance

Audited 2026-11-22 before training. Dataset: [bihungba1101/slam-en-es-knowledge-tracing](https://huggingface.co/datasets/bihungba1101/slam-en-es-knowledge-tracing/tree/44c6826ebbfb7ac7f0af5d181b5f1a65cf4f5304), revision **44c6826ebbfb7ac7f0af5d181b5f1a65cf4f5304**. The local range cache holds the pinned card, LICENSE.md, CITING.md and converter; [provenance.json](provenance.json) records their SHA256 hashes, observed Arrow schemas, source row counts, ordering diagnostics, preprocessing config and subset fingerprint.

The card's metadata says `license: other`. Its [license file](https://huggingface.co/datasets/bihungba1101/slam-en-es-knowledge-tracing/blob/44c6826ebbfb7ac7f0af5d181b5f1a65cf4f5304/LICENSE.md) explicitly specifies **CC BY-NC 4.0**, copyright (c) 2016 Duolingo, Inc. Preserve attribution, license link and notices, mark modifications and restrict use to noncommercial purposes. Data and derivatives are supplied without warranties. This experiment retains raw/prepared records and weights locally under ignored directories. The changed learner splits, features and aggregate analyses are documented here. [License terms](https://creativecommons.org/licenses/by-nc/4.0/).

Required source citation: B. Settles, C. Brust, E. Gustafson, M. Hagiwara and N. Madnani (2018), *Second Language Acquisition Modeling*, BEA/NAACL-HLT, ACL. [Source data DOI](https://doi.org/10.7910/DVN/8SWHNO); [paper](https://aclanthology.org/W18-0506/), DOI 10.18653/v1/W18-0506. Credit also the Hugging Face conversion by bihungba1101. No endorsement is implied.

## Field contract

The [pinned converter](https://huggingface.co/datasets/bihungba1101/slam-en-es-knowledge-tracing/blob/44c6826ebbfb7ac7f0af5d181b5f1a65cf4f5304/convert_to_hf.py) defines these Arrow fields and derivations. H = historical input, C = permitted target context, A = audit/grouping only, X = excluded.

| Field | Arrow type | Meaning | Use |
|---|---|---|---|
| user_id | string | Anonymous learner key | A |
| interaction_index | int32 | Learner attempt number within source split | A |
| exercise_id | string | First reference token ID minus final two characters | X |
| prompt | string | Presented text, if supplied | X |
| country | string | Source learner country code(s) | X |
| client | string | Device/platform | X |
| session | string | Lesson/practice/test category | H/C |
| format | string | Exercise interface type | H/C |
| days | float64 | Days since this learner began the course | H/A |
| response_time_seconds | float64 | Submission duration; negative/null becomes null | H |
| token_ids | list[string] | Aligned reference token identifiers | X |
| reference_tokens | list[string] | Closest acceptable reference, selected through submission alignment | X |
| pos_tags | list[string] | Reference universal POS annotations | X |
| morphological_features | list[string] | Reference morphology/fine POS | X |
| dependency_relations | list[string] | Reference syntactic labels | X |
| dependency_heads | list[int32] | One-based syntactic heads; zero is root | X |
| token_labels | list[int8] | Aligned mistakes: zero OK, one error | X |
| token_accuracy | float32 | One minus mean token mistake label | X |
| correct | int8 | One iff all token labels are zero | H/target |
| prompt_available | bool | Whether prompt is non-null | X |

All token lists align by position. Actual learner response text is unavailable. The model uses the stored interaction-level `correct`; it does not recalculate or reinterpret that label. Reference-derived identifiers and annotations are excluded even if they look like exercise metadata.

## Pre-answer availability and ordering evidence

The [source paper, sections 2.3–2.5 and Figure 1](https://sharedtask.duolingo.com/papers/settles.slam18.pdf), describes the exercise interfaces and learner-selected lesson/practice sessions. Their categories precede the response; these are the only allowed target fields. Recorded response duration necessarily follows answering. The exact instant represented by `days` is insufficiently established for a target-time feature, so target days is excluded.

The paper establishes that each learner's first 80% of exercises are source TRAIN, subsequent 10% DEV, final 10% TEST. The converter emits records in file order and restarts its learner counter for each source file. Reconstruction concatenates original source train → validation → test, preserving original file order. It then **checks**, rather than sorts on, nondecreasing days and increasing split-local indices. Ties retain source order, not an invented timestamp ordering. Reversed time, duplicate source keys or reversed partitions raise errors. Each local event retains `source_partition`, absolute `source_row`, split-local `interaction_index`, and reconstructed `sequence_index` for auditing.

## Measured pilot audit

All three sources contain 1,054,368 interactions across a union of 2,593 learners (824,012 / 115,770 / 114,586 rows). Only learner-ID columns are scanned universally; permitted columns are then loaded from row groups intersecting the selected learners. Unselected records are discarded immediately. No text, aligned tokens or linguistic arrays are loaded for modeling.

The sample is the 100 smallest SHA256 values of `sample:42:<original learner ID>`. A separate `split:42:<ID>` ordering assigns 70/15/15 learners. No learner is replaced or filtered based on outcomes. Every selected learner's source interactions stay together. The subset fingerprint in provenance allows deterministic identity verification without publishing IDs.

| Partition | Learners | Interactions | Prediction targets | Positive target rate | Missing response time | Learner span days min/median/max |
|---|---:|---:|---:|---:|---:|---|
| Train | 70 | 29,706 | 29,636 | 0.710521 | 575 | 2.023 / 15.301 / 28.679 |
| Validation | 15 | 5,203 | 5,188 | 0.692945 | 1 | 2.952 / 15.352 / 27.940 |
| Test | 15 | 6,943 | 6,928 | 0.721853 | 1 | 4.181 / 18.116 / 29.166 |

Other modeled fields have zero missing values in this subset. Nonmodeled columns are schema-audited; their missingness is not exhaustively scanned. Relative day ranges are train 0.003–29.751, validation 0.004–28.188, test 0.003–29.169, **not calendar coverage**. Absolute collection dates cannot be reconstructed from these fields.

Ordering: 200 cross-partition learner transitions checked; 29,909 consecutive equal-day transitions preserved; zero split-local index gaps, duplicates or detected time reversals. This validates compatibility with documented order, not independent recovery of exact within-timestamp wall-clock instants.

Preparation read 3,527,852 Parquet bytes over the network versus 40,122,310 total Parquet bytes, plus approximately 34 KB of provenance documents. Sampled peak process RSS was 104.34 MiB; elapsed projected acquisition/preparation was 482.18 seconds. Range granularity can include small amounts of unused trailing data. Full records for all learners were neither downloaded nor materialized. Offline reconstruction subsequently used zero network bytes and reproduced all three split-file hashes.

## Known limitations and discrepancies

- The card describes `days` as starting Spanish; the official `en_es` track and Spanish-prompt/English-reference examples identify **English learning by Spanish speakers**. The converter's CLI default prefix is `es_en`, which is another reason not to regenerate blindly using its defaults. The published Parquet revision, not an unaudited default invocation, is the experiment's source of truth. The original upstream files were not redownloaded to independently verify every conversion.
- The source cohort is a selected group of progressing beginners during their first month, from an older Duolingo product. It cannot represent all students, languages, course stages, dropouts or modern exercise selection. Prior learning and personalized exercise assignment confound observational patterns. See the paper's corpus collection section.
- Whole-exercise strict correctness depends on reference alignment and exercise length. Token errors ignore some orthographic distinctions; the observed label is not a comprehensive assessment of knowledge. The benchmark makes a more conservative feature choice than the original token-level shared task and is not directly comparable to its leaderboard.
- Coarse relative days produce many ties and zero gaps. Negative response durations were already normalized to missing by the conversion; their original counts are not recoverable. Large imbalance in response-time missingness across the sampled learner partitions is retained and reported, not treated as zero.
- The new split measures generalization to unseen learners with online observed outcomes. It does not test cold-start first-attempt predictions, future cohorts, interventions or accountability policies. Overlapping within-learner windows require learner-level uncertainty; 15 held-out learners give a small pilot, not confirmatory evidence.

See [README](../README.md) for feature provenance, frozen config, artifacts, masks, ablations and reproduction commands.
