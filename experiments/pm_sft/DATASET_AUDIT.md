# Dataset audit

## Sources

| Source | License and revision | Inspected schema | Role and limitations |
| --- | --- | --- | --- |
| Amazon Science MASSIVE | CC BY 4.0. Hugging Face revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`. Amazon archive release 1.0. | A JSONL file per locale. Raw records have `id`, `locale`, `partition`, `scenario`, `intent`, `utt`, `annot_utt` and `worker_id`. `annot_utt` marks slots inline. | Parallel single-turn, single-intent voice assistant utterances localized from SLURP. It lacks PM progress semantics, multi-intent ground truth, pending proposal state and adjudicated uncertainty. Source locale wording can differ in meaning. |
| NVIDIA Nemotron-Personas-Vietnam | CC BY 4.0. Revision `950724d49b7b41eee7925ae035a2059286c890ff`. | Vietnamese Parquet with UUID, six persona narrative fields, six culture/skills/hobbies/ambition fields and eight demographic/geographic fields. | Synthetic profiles informed varied work, sports, arts and culinary situations. No persona text or metadata enters the corpus. Persona descriptions supply neither conversation labels nor evidence for how individuals behave. |

MASSIVE attribution goes to Jack FitzGerald and colleagues at Amazon Science. The experiment preserves selected utterance strings, remaps their intents and normalizes two annotated relative-day slots. NVIDIA attribution is for the documented categories that informed fictional scenario construction. [MASSIVE paper](https://aclanthology.org/2023.acl-long.235/), [MASSIVE dataset card](https://huggingface.co/datasets/AmazonScience/massive), [MASSIVE archive and schema](https://github.com/alexa/massive), [NVIDIA pinned dataset card](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam/tree/950724d49b7b41eee7925ae035a2059286c890ff), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### MASSIVE mapping

- The selected `general_greet` IDs are 45, 166, 719, 1100 and 1921. They map to `GENERAL_CONVERSATION`. The annotated date for 1100 maps to `today`.
- Calendar query 1274 retains the `tomorrow` date and maps to `CLARIFY` with `request` missing because confirming a meeting cancellation is outside PM. Meeting names, identities, worker IDs and unrelated labels are omitted.
- The six aligned families come from the original train partition and remain together in PM training. English uses `en-US`, Vietnamese uses `vi-VN` and Korean uses `ko-KR`. Exact normalized-text duplicates across partitions, repeated IDs and family leakage are rejected.

### NVIDIA schema

The identifier is `uuid`. The six narratives are `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona` and `persona`. The attribute fields are `cultural_background`, `skills_and_expertise`, `skills_and_expertise_list`, `hobbies_and_interests`, `hobbies_and_interests_list` and `career_goals_and_ambitions`. Context fields are `sex`, `age` (integer), `marital_status`, `education_level`, `occupation`, `zone`, `region` and `country`. The card describes one million generated profiles. Demographic frequencies can shape generated stereotypes, and the fields supply no coverage for English or Korean conversations. [NVIDIA schema](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam/blob/950724d49b7b41eee7925ae035a2059286c890ff/README.md)

## Corpus and review

| Split | Families | EN | VI | KO | PM authored | MASSIVE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 32 | 32 | 32 | 32 | 78 | 18 |
| Validation | 6 | 6 | 6 | 6 | 18 | 0 |
| Holdout | 10 | 10 | 10 | 10 | 30 | 0 |

The author checked the intent meaning, number roles, dates, missing fields, proposal context and localized label alignment while constructing the source. The preparation audit verified schema validity, gold-policy consistency, family membership and aligned language counts. There was no independent bilingual adjudication. The corpus covers mixed number roles, relative days, conflicting durations, ambiguous references, colloquial phrasing, approvals with no valid pending proposal, corrections and unsupported requests. All authored text is fictional. There are no Discord logs or SQLite records.

The dataset deliberately tests a small vocabulary of PM slots, single-valued slots and recognizable approval IDs. The 30 scored holdout examples represent 10 independent scenarios with three correlated language variants each. Use the local `manifest.json` for label counts and reproducibility fingerprints.
