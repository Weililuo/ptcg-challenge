# PTCG-Challenge: Autonomous Battle Agent & Iterative DAgger Pipeline

An end-to-end imitation learning framework for the Pokémon Trading Card Game (PTCG), built on the CABT environment.

This repository implements a full-stack competitive AI architecture spanning raw tournament replay parsing, tabular feature engineering, XGBoost behavioral cloning, deterministic heuristic sub-decision engines, and a reward-weighted DAgger (Dataset Aggregation) self-play loop with a statistically gated arena promotion step.

---

## System Architecture

The battle agent addresses dynamic action-space branching and non-transitive matchup dynamics through a multi-stage training pipeline:

```text
[Raw CABT Replays] ──> [dataset_builder.py: Feature Extraction (21-D)] ──> [train_bc_model.py: XGBoost BC]
                                                                                        │
                                                                                        ▼
[ai.py Heuristic Sub-Decisions] ──> [ai.py Inference Engine] <──────────────────────────┘
                                            │
                                            ▼
                          [rollout_worker.py: Parallel Self-Play Rollout]
                                            │
                                            ▼
                          [dagger_common.py: Reward-Weighted Sample Extraction]
                                            │
                                            ▼
                          [retrain_dagger.py: Warm-Start Booster Update]
                                            │
                                            ▼
                          [arena_judge.py: Gated Champion vs. Candidate Arena]
                                            │
                                            ▼
                          [run_dagger_loop.py: Iterative Orchestration]
```

---

## Core Pipeline Components

### 1. Data Parsing & Feature Engineering (`dataset_builder.py`)

- **Replay Ingestion**: Traverses raw tournament JSON logs under `datasets/` to extract main-phase trajectories.
- **Tabular Feature Matrix**: Flattens game observations into structured vectors written to `train_features.csv` (~4.42M rows). The canonical feature ordering is defined once in `ai.py` / `dagger_common.py` as `FEATURE_NAMES` (**21 dimensions**):

  | Group | Features |
  | :--- | :--- |
  | Turn context | `turn`, `turn_action_count`, `options_count` |
  | Hand resources | `hand_size`, `hand_grass_cnt`, `hand_lightning_cnt`, `hand_fight_cnt` |
  | Board state | `is_active_raging_bolt`, `active_hp_ratio`, `active_energy_cnt`, `has_latias_ex`, `bench_ogerpon_cnt`, `total_board_energies`, `my_bench_lowest_hp` |
  | Prize race | `my_prizes_remaining`, `opp_prizes_remaining`, `prize_diff` |
  | Opponent reads | `opp_active_immune_ex`, `opp_has_crustle`, `opp_has_dragapult`, `opp_has_froslass` |

  `rollout_worker.py` asserts at runtime that `ai.py` and `dagger_common.py` agree on `FEATURE_NAMES`, raising `RuntimeError` on drift.

  > **Schema warning**: the checked-in `train_features.csv` predates the current schema — its header carries only the first 17 of these columns (it lacks `my_bench_lowest_hp`, `opp_has_crustle`, `opp_has_dragapult`, `opp_has_froslass`). Both shipped boosters have `num_feature = 21`. Re-running `dataset_builder.py` is required before `train_bc_model.py` can produce a model the inference engine accepts; a 17-feature booster makes `inplace_predict` raise, which `ai.py` swallows by falling back to `MAIN_FALLBACK_SCORES`.

### 2. Behavioral Cloning Pre-Training (`train_bc_model.py`)

- **Framework**: Multi-class gradient boosted decision trees via XGBoost (`tree_method='hist'`) — chosen for fast convergence and to avoid the optimization instability of deep actor-critic networks in discrete card state spaces.
- **Optimization Setup** (hardcoded; the script takes **no command-line arguments** and resolves `train_features.csv` / `xgb_model.json` / `action_label_mapping.json` from absolute paths at the repository root):

  | Hyperparameter | Value |
  | :--- | :--- |
  | `objective` | `multi:softprob` |
  | `eval_metric` | `mlogloss` |
  | `n_estimators` | `250` |
  | `max_depth` | `7` |
  | `subsample` / `colsample_bytree` | `0.85` / `0.85` |
  | `learning_rate` | `0.08` |
  | `test_size` (stratified) | `0.2` |

- **Action Space Mapping**: Drops long-tail labels with fewer than 50 occurrences, then writes the surviving label set to `action_label_mapping.json` as `{"class_to_option": [...]}` — this is the class-index → option-index table `ai.py` uses to align booster outputs with `select.option` positions.
- **Validation Performance** (as previously reported for the full-data BC run): 36.74% Top-1 / 68.71% Top-3 accuracy over an 883,506-row validation split (vs. a 2.27% random baseline), with train loss 1.873 vs. val loss 1.895. Note that the two boosters currently checked in are DAgger outputs (see below), not artifacts of this script.

### 3. Inference & Rule-Based Tactical Sub-Decisions (`ai.py`)

`ai.py` is a single self-contained agent module exporting `make_ai_agent(deck, player_index)` → `AIAgent`, whose `__call__` returns `select_legal_options(request, score_options(request, observation))`.

- **Model Loading**: The booster is built at **import time** (not lazily), because a first-decision lazy load would exceed the environment's `actTimeout` of 1 s. `PTCG_MODEL_PATH` / `PTCG_MAPPING_PATH` override the default `xgb_model.json` / `action_label_mapping.json` so `arena_judge.py` can hold a champion and a candidate in one process. If the native XGBoost call fails, a pure-Python tree walker over the model JSON takes over; if that fails too, scoring degrades to the constant fallback table `{7: 1.5, 8: 4.0, 9: 2.0, 13: 6.0, 14: 2.0}`.

- **Main-Phase Scoring (select type 0)**: base score per option is the booster's class probability, remapped through `class_to_option`; options absent from the mapping score 0.0. Rule-based adjustments are then added on top:

  | Trigger | Adjustment |
  | :--- | ---: |
  | Early game (`turn <= 2`), no Raging Bolt ex on board, playing Ultra Ball / Cheren / Bug Catching Set / Pokégear 3.0 | `+120.0` |
  | Same gate, playing Lillie's Determination / Crispin | `+90.0` |
  | Iron Leaves ex, with ≥ 2 grass energy on board | `+80.0` |
  | Iron Leaves ex, with < 2 grass energy on board | `−40.0` |
  | Energy Retrieval, with ≥ 2 basic energies in discard | `+85.0` |
  | Energy Retrieval, with < 2 in discard | `−20.0` |
  | Unfair Stamp (opponent hand −3, behind on prizes, own KO this turn) | `+130.0` |
  | Unfair Stamp, trigger conditions unmet | `−80.0` |
  | Boss's Orders, opponent active immune to ex, no Bloodmoon in hand/bench | `+150.0` |
  | Bloodmoon Ursaluna (135), opponent active immune to ex | `+85.0` |
  | Bloodmoon Ursaluna ex (44), opponent at ≤ 2 prizes | `+60.0` |
  | Switch, when Latias ex is on board (free retreat already available) | `−25.0` |
  | Retreat/change (type 9), opponent immune to ex and Bloodmoon Ursaluna benched | `+90.0` |
  | Retreat/change (type 9), Latias ex on board | `+30.0` |

  Energy attachment (type 8) is scored by `_attach_score()`, which hard-refuses support Pokémon (`AUXILIARY_POKEMON_IDS`: Fezandipiti ex, Latias ex) at `−200.0` and otherwise applies a per-energy, per-target table: Lightning or Fighting onto Raging Bolt ex (`120.0` if the type is missing, `70.0` if already present), Grass onto Iron Leaves ex (`110.0`) or Teal Mask Ogerpon ex (`85.0`), Fighting onto Bloodmoon Ursaluna (`90.0`), and a generic default table (`Raging Bolt ex 100.0 > Iron Leaves ex 80.0 > Ogerpon ex 75.0 > Bloodmoon 60.0 > Bloodmoon ex 30.0`).

- **Sub-Decision Routing (select type ≠ 0)**: deterministic tables replace model output entirely, eliminating exploratory degradation inside multi-step card effects:

  | Select context | Behaviour |
  | :--- | :--- |
  | `type 1, context "7"`, discard area present | Recovery tiers — Night Stretcher: Raging Bolt ex `110.0` when absent from hand+bench, otherwise Bloodmoon ex / Bloodmoon `100.0` > Raging Bolt ex `80.0` > Iron Leaves ex `60.0`. Energy Retrieval: Grass `50.0` > Lightning `35.0` = Fighting `35.0`. |
  | `type 1, context "7"`, looking area present | Search tiers — Pokégear 3.0 reveals: Cheren `100.0` > Crispin `80.0` > Lillie's Determination `60.0` > Boss's Orders `40.0`. Bug Catching Set reveals: Iron Leaves ex `80.0` > Teal Mask Ogerpon ex `60.0` > Basic Grass Energy `50.0`. Unknown cards default to `10.0` / `5.0`. Cards revealed from the opponent's deck score `0.0`. |
  | `type 1, context "8"` | Discard cost — key items protected at `−200.0`; basic energy `100.0` > Cheren `80.0` > Crispin `65.0` > Lillie `45.0`; Pokémon `40.0` if duplicated in hand else `−50.0`. |
  | `type 1, context "3"`, trigger Switch | Retreat target — against an ex-immune active: Bloodmoon Ursaluna `100.0`, else `10.0`. Otherwise: Raging Bolt ex with ≥ 2 energy `90.0`, Iron Leaves ex with ≥ 3 energy `70.0`, fallback `20.0`. |
  | `type 2, context "26"` | Energy discard — Ogerpon ex `120.0` > Iron Leaves ex `80.0` > support `70.0` > Bloodmoon `60.0`; Raging Bolt ex is shielded at `−80.0` while active with ≤ 2 energy. |
  | `type 4` | Attach sub-select, delegated to the same `_attach_score()` table as main-phase attachments. |

### 4. Distributed DAgger Self-Play Loop (`run_dagger_loop.py`)

- **Multi-Process Concurrency**: `rollout_worker.py` dispatches matches through a `multiprocessing` **spawn** pool (`--workers`, default 16), avoiding nested lock contention in runtime loaders. `run_dagger_loop.py` wraps the three stages of each iteration with its own worker budget (`--workers`, default 6).
- **Opponent Curriculum**: each round samples opponents from a pool of the mirror deck plus the expert archetypes under `ptcg-ai/decks/` (`--expert-frac`, default 0.5). Sampling weights are adaptive: base weight 0.20 for the mirror deck vs. 0.10 for each expert deck, rescaled by how far each opponent's recent win rate sits from the target, so under-performing matchups are oversampled.
- **Reward-Weighted Filtering** (`dagger_common.compute_sample_weight`): only winning trajectories (`reward == 1`) are recorded, and the weighting is **per-match**, not per-decision. A match's total sample mass is `clip(final_score, w_min, w_max)` — `w_min = 0.25`, `w_max = 5.0` by default — divided evenly across that match's recorded rows, so a decisive win contributes at most 20× the mass of a marginal one. `--weight-mode flat` disables the scaling, and `--score-floor` drops low-scoring wins entirely. `--record-sides both` also mines the opponent's winning trajectory.
- **Warm-Start Incremental Retraining** (`retrain_dagger.py`): continues the existing booster rather than restarting, adding `--boost-rounds` new trees per class (`25` by default; the loop passes `50`) at `--lr 0.01`, mixing in historical rows via `--history-frac 0.5` / `--history-weight 1.0` plus an optional `selfplay/baseline_promoted` corpus. `--max-depth` must match the base booster's structure. Output lands in `candidate_xgb_model.json` + `candidate_action_label_mapping.json`.
- **Arena Promotion Gate** (`arena_judge.py`): plays candidate against champion under two env-aliased module loads, additionally gating on win rate against the expert archetype decks. A round is inconclusive below `--min-valid-frac 0.8` of valid games; promotion requires clearing both `--threshold` (default `0.55`, or `0.57` as invoked by the loop) and `--min-expert-winrate` (default `0.35`, or `0.30` from the loop). Swap happens on promotion only if `--promote` is passed; rejected candidates can be archived with `--archive-rejected`.

### 5. Orchestration (`run_dagger_loop.py`)

Drives `rollout_worker.py` → `retrain_dagger.py` → `arena_judge.py` for `--rounds` iterations (`10` by default, `200` matches per round), logging each round's decision to `selfplay/loop_log.jsonl` and optionally re-rendering plots with `--viz`.

---

## Deck Strategy: Raging Bolt ex Toolbox

The agent pilots an aggressive, high-tempo Raging Bolt ex list built for rapid resource acceleration and multi-prize knockouts. `deck.csv` stores raw card IDs, one per line; the composition below is derived from those IDs and totals exactly 60 cards.

**Card List (60 Cards)**

```text
├── Pokémon (12)
│   ├── 3x Raging Bolt ex (id 63, Primary Attacker)
│   ├── 4x Teal Mask Ogerpon ex (id 96, Draw Engine & Grass Acceleration)
│   ├── 1x Bloodmoon Ursaluna ex (id 44, Secondary ex Attacker)
│   ├── 1x Bloodmoon Ursaluna (id 135, Non-ex Attacker)
│   ├── 1x Fezandipiti ex (id 140, Flip the Script Draw Support)
│   ├── 1x Latias ex (id 184, Skylight Free-Retreat Pivot)
│   └── 1x Iron Leaves ex (id 27, Grass Attacker)
├── Trainers (28)
│   ├── 4x Ultra Ball (1121), 4x Lillie's Determination (1227)
│   ├── 3x Crispin (1198), 2x Cheren (1224), 2x Boss's Orders (1182)
│   ├── 2x Pokégear 3.0 (1122), 2x Bug Catching Set (1094)
│   ├── 2x Energy Retrieval (1118), 2x Energy Search (1119)
│   ├── 2x Switch (1123), 2x Night Stretcher (1097)
│   └── 1x Unfair Stamp (1080)
└── Energy (20)
    ├── 10x Basic Grass Energy (id 1)
    ├── 4x Basic Lightning Energy (id 4)
    └── 6x Basic Fighting Energy (id 6)
```

> **Note**: The list carries its own non-ex attacker line (Bloodmoon Ursaluna plus Iron Leaves ex) so the deck is not shut down by damage-immune abilities — the counter-play the Main-Phase heuristics above key off (`opp_active_immune_ex`).

---

## Repository Structure

```text
.
├── ai.py                               # Inference agent: 21-D extractor + XGBoost + heuristics
├── dagger_common.py                    # Shared paths, FEATURE_NAMES, sample weighting, IO helpers
├── dataset_builder.py                  # Raw replay parser generating train_features.csv
├── train_bc_model.py                   # Behavioral cloning entrypoint (no CLI args)
├── rollout_worker.py                   # Parallel self-play rollout producing rollout_rows.csv
├── retrain_dagger.py                   # Warm-start booster update producing a candidate model
├── arena_judge.py                      # Champion/candidate/expert arena & promotion gate
├── run_dagger_loop.py                  # Main orchestrator for the iterative DAgger loop
├── run_match.py                        # Single match simulation runner (no CLI args)
├── run_rl_pipeline.py                  # Standalone multiprocess self-play runner
├── visualize_pipeline.py               # Win-rate trend & matchup plotting
├── human.py                            # Human-input agent to play against the AI agent
├── deck.csv                            # Primary 60-card Raging Bolt ex list (raw card IDs)
├── action_label_mapping.json           # class_to_option mapping of the deployed model
├── xgb_model.json                      # Deployed booster (21 features x 44 classes, 4400 trees)
├── candidate_xgb_model.json            # Latest DAgger candidate (6600 trees)
├── candidate_action_label_mapping.json # class_to_option mapping of the candidate
├── train_features.csv                  # BC dataset (~4.42M rows, legacy 17-feature schema)
├── results.csv                         # Match outcome log (match_id, won, raw_score, final_score)
├── ptcg-ai/                            # Expert archetype decks & PyTorch training toolkit
│   ├── decks/                          # 8 archetypes: alakazam, crustle, dragapult, froslass,
│   │                                   #   hydrapple, mega_lopunny, raging_bolt, slowking
│   └── models/                         # Published expert checkpoints (base, experts, old)
├── datasets/                           # Raw CABT tournament replay JSON
└── selfplay/                           # Round artifacts: rollout_rows.csv, arena, retrain, plots
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n ptcg python=3.10 -y
conda activate ptcg
pip install xgboost scikit-learn pandas numpy matplotlib seaborn kaggle-environments
```

### 2. Build the Feature Dataset

Re-parse the raw replays. Note that `dataset_builder.py` hardcodes its input and output paths:

```bash
python dataset_builder.py
```

### 3. Single-Match Simulation

`run_match.py` takes no arguments and reads `deck.csv` from the repository root:

```bash
python run_match.py
```

### 4. Supervised BC Training

`train_bc_model.py` also takes no arguments — it reads `train_features.csv`, then writes `xgb_model.json` and `action_label_mapping.json`:

```bash
python train_bc_model.py
```

### 5. Iterative DAgger Self-Play Loop

Run the full rollout → retrain → arena cycle and swap the deployed model when a candidate clears the gate:

```bash
python run_dagger_loop.py \
  --rounds 10 \
  --matches 200 \
  --workers 6 \
  --boost-rounds 50 \
  --lr 0.01 \
  --threshold 0.57 \
  --min-expert-winrate 0.30 \
  --promote
```

Individual stages can be driven directly, e.g. `python rollout_worker.py --matches 1000 --workers 16`, `python retrain_dagger.py --round 16 --new-rows selfplay/round_016/rollout_rows.csv`, or `python arena_judge.py --round 16 --games 100`.

### 6. Generate Evaluation Analytics

Render diagnostic win-rate trends and opponent matchup radars:

```bash
python visualize_pipeline.py \
  --out-dir ./selfplay \
  --arena-dir ./selfplay/arena \
  --plots-dir ./selfplay/plots
```

---

## References

- Foster, D. J., Block, A., & Misra, D. (2024). *Is behavior cloning all you need? Understanding horizon in imitation learning* (arXiv:2407.15007v2). arXiv. https://doi.org/10.48550/arXiv.2407.15007
- Hua, D., Sun, Y., Huang, R., Gao, F., Wang, C., & Yang, Y. (2026). *PTCG-Bench: Can LLM agents master Pokémon Trading Card Game?* (arXiv:2605.29653v1). arXiv. https://doi.org/10.48550/arXiv.2605.29653
