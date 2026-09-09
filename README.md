# binder-adapter

Cohesive integration between **Jnana** (`main`) and **StructBioReasoner** (`pydantic_refactor`)
that turns Jnana's ProtoGnosis tournament into a multi-objective protein-binder
design campaign engine.

Neither repository is modified. The adapter integrates through versioned
schemas, a subprocess scoring worker, and a runtime monkey-patch.

```
Jnana generation ──► CandidateArtifact ──► mutation-budget gate ──► SBR scoring
                                                │ reject                │ shards (target × batch)
                                                ▼                       ▼
                                          resample request      ObjectiveReport
                                                                        │
                                                              Pareto + diversity
                                                                        ▼
                                                                 RankingDecision
```

## Status of the five objectives

The adapter never invents a number. Each objective is either measured from real
StructBioReasoner code or explicitly marked unavailable with a reason that
travels in the output.

| Objective | Status | Source |
|---|---|---|
| `developability` | **measured** | `computational_design.quality_control.SequenceQualityControl` — fraction of its 8 filters passed |
| `binding_affinity` | **measured when a complex PDB is supplied** | default: `computational_design.energy.SimpleEnergy` (chain A/B contacts within 5.0 Å). With `md_config`: physics-based MM-GBSA ΔG_bind from an OpenMM trajectory (see below) |
| `specificity_off_target_proxy` | **measured** | `1 − max_identity` against an off-target panel (adapter-side; the embedding route needs `genslm_esm`, which is not installed) |
| `thermostability` | **measured with `md_config` + a complex PDB** | conformational-stability proxy (backbone RMSF + Rg drift + energy variance) from an OpenMM implicit-solvent trajectory. Unavailable otherwise (MD is opt-in) |
| `structure_confidence` | unavailable | `ChaiAgent` needs GPU + Chai-1 weights; `TrajectoryAnalysisAgent._calculate_confidence` is dead code (a `return 0.75` stub shadows the real RMSD heuristic) |

Only objectives available for **every** candidate enter the Pareto comparison —
a partially-measurable objective would bias against candidates that lack it.
The objectives actually used are reported in `active_objectives`.

### Why not the StructBioReasoner CLI

`struct_bio_reasoner.py --mode batch` does not run on `pydantic_refactor`:

```
ImportError: cannot import name 'ProteinEngineeringSystem' from 'struct_bio_reasoner'
```

`ProteinEngineeringSystem` does not exist anywhere on the branch, `core/` holds
only stale `__pycache__`, and `run.sh` points at a `binder_design_reasoner.py`
that is likewise absent. Both are leftovers from the `main` lineage.

The adapter therefore runs its own scoring worker
(`binder_adapter/sbr_backends/sbr_scoring_worker.py`), which imports the SBR
modules that do work. That worker is a thin shell over the shared scoring core
(`binder_adapter/sbr_backends/scoring.py`), the single source of truth both
scoring tiers call — so a subprocess report and an in-process report for the
same candidate are byte-for-byte identical, with no second copy to drift.

## Scoring backends (Tier 1 / Tier 2)

Both backends produce the same `ObjectiveReport`s; pick with `--backend`
(CLI) or `AdapterConfig(backend=...)`.

| Backend | `--backend` | How it scores | Use when |
|---|---|---|---|
| `SubprocessScoringBackend` | `subprocess` (default) | one subprocess per `(target, shard)` running the worker | crash-prone/untrusted scoring; process isolation; Slurm/PBS array unit-of-work |
| `InProcessScoringBackend` | `in_process` | calls `scoring.score_shard` directly in the host interpreter | host already imports SBR and you want to skip per-shard process spawn + JSON round-trip |

`test_inprocess_backend.py::test_subprocess_and_in_process_reports_are_identical`
holds the two tiers to parity — objective scores, availability, reasons and
evidence must match across backends. That test is what makes the shared core a
guarantee rather than a hope.

## OpenMM MD objectives (physics-based binding + thermostability)

StructBioReasoner's MD package (`agents/molecular_dynamics/{MD,distributed,mmpbsa_agent}.py`)
is Parsl **orchestration only** — its actual physics lives in the external
`molecular_simulations` package, which shells out to a full Amber install
(`MMPBSA.py`/`sander`/`pmemd`), needs `numpy>=2` (conflicts with the working
env's 1.26), and expects precomputed MD trajectories. So that path stays dark.

Instead the adapter ships a self-contained OpenMM engine
(`binder_adapter/sbr_backends/md_engine.py`) — **no Amber, no molecular_simulations,
no numpy-2 conflict**. It uses OpenMM's bundled amber14 ff14SB force field and GB
implicit solvent (`gbn2`) plus `pdbfixer` for structure prep. From one short
implicit-solvent trajectory of a complex PDB it derives two objectives:

- **binding_affinity → MM-GBSA ΔG_bind** (single-trajectory): `<E_complex> −
  <E_receptor> − <E_ligand>`, averaged over frames, chains split by id. This
  upgrades `binding_affinity` from the crude SimpleEnergy contact count to a real
  free energy (and overrides it only when the MD run actually succeeds).
- **thermostability → conformational-stability proxy**: backbone RMSF + radius-of-
  gyration drift + potential-energy variance over the trajectory. Explicitly a
  proxy, **not** a folding ΔΔG/ΔTm (which needs alchemical FEP) — labelled as such
  in the evidence.

MD is **opt-in**. Add `md_config` to the campaign context (`{}` for demo defaults)
and supply a multi-chain `complex_pdb_path`:

```json
{
  "framework_sequence": "…",
  "target_ids": ["MDM2"],
  "complex_pdb_path": "complex.pdb",
  "md_config": {"equil_steps": 500, "prod_steps": 2000, "sample_interval": 200, "platform": "CPU"}
}
```

Defaults are demo-scale (picoseconds, CPU) so the pipeline is provable end-to-end
in seconds. For production, raise `equil_steps`/`prod_steps` to 1e5–1e6 and set
`"platform": "CUDA"`. A failed simulation never fabricates a score — the objective
falls back to unavailable with the reason, and (for binding) to SimpleEnergy.

Requires `openmm` and `pdbfixer` (the latter installs cleanly against the
existing numpy 1.26 / openmm 8.3). MD tests skip automatically when either is
absent.

### Per-candidate mutation threading

Each candidate is simulated as its *own* mutant, not the wild-type template. When
`md_config` names a `binder_chain`, the candidate's substitutions (diffed against
the framework) are threaded onto that chain with `PDBFixer.applyMutations` before
the system is built, so the MM-GBSA ΔG and stability proxy reflect the actual
mutations. The applied mutation list (e.g. `["GLU-3-CYS", "PHE-5-TRP", ...]`) and
count travel in the binding evidence. Mutant side chains can be clashy, so the
engine re-minimizes hard and equilibrates in small chunks with a NaN-guarded
retry; a candidate that still can't be stabilized is reported unavailable (never
scored from a broken trajectory).

### Realistic end-to-end demo

`examples/run_real_campaign.py` runs the whole pipeline on a real crystal
structure. It fetches 1YCR (MDM2 receptor + p53 peptide), treats the p53 chain as
the binder framework, generates in-budget point-mutant candidates, threads each
onto the structure, and runs the MD campaign to a Pareto ranking:

```bash
python examples/run_real_campaign.py --n-candidates 4 --prod-steps 1500
# -> ranked survivors with binding(-dG), thermo, dev per candidate;
#    full JSON (incl. applied mutations + provenance) in runs/real_campaign/
```

A representative run (4 candidates, ~15-residue binder, CPU, demo scale) lights up
all four measurable objectives — `binding_affinity`, `thermostability`,
`developability`, `specificity_off_target_proxy` — with distinct threaded
mutations and distinct ΔG per candidate; only `structure_confidence` stays
unavailable.

## Mutation budget (reject and resample)

`mutated_fraction = positions differing from framework / len(framework)`,
constrained to `[0.20, 0.50]`, enforced as **both** the fraction and the derived
counts `[ceil(0.20·L), floor(0.50·L)]`. Violations are rejected, never repaired.

`L` is computed per campaign. Rejected candidates appear in
`invalid_candidates` with the exact constraint tripped, and
`resample_shortfall` tells the generation layer how many replacements to draw.

## Usage

### Front-end 1 — adapter CLI

```bash
PYTHONPATH=. python -m binder_adapter.cli \
  --campaign-context campaign.json \
  --hypotheses hypotheses.json \
  --run-id demo \
  --shard-size 16 \
  --max-survivors 10 \
  --artifact-dir ./runs \
  --sbr-root /Users/ramanathana/Work/StructBioReasoner
```

`campaign.json`:

```json
{
  "framework_sequence": "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ",
  "target_ids": ["TARGET_A", "TARGET_B"],
  "antigen_structure_ref": "target.pdb",
  "complex_pdb_path": "",
  "specificity_reference_sequences": ["MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"]
}
```

Supplying `complex_pdb_path` (per campaign or per hypothesis) is what promotes
`binding_affinity` from unavailable to measured.

### Front-end 2 — Jnana calls the adapter

```python
from binder_adapter.jnana_hook import patch_ranking_agent, get_last_resample_request

restore = patch_ranking_agent(
    campaign_context={"framework_sequence": FW, "target_ids": ["TARGET_A"]},
    sbr_root="/Users/ramanathana/Work/StructBioReasoner",
)

# ... run the ProtoGnosis tournament ...

req = get_last_resample_request()
if req["needs_resample"]:
    generate_more(req["shortfall"])   # reject-and-resample

restore()
```

Returned rankings keep Jnana's existing keys (`hypothesis_id`, `rank`, `score`,
`justification`) and add `pareto_front_id`, `active_objectives`, and full
`objective_reports`. If the adapter raises, the hook logs and falls back to
Jnana's original ranking so a tournament never dies on an adapter fault.

## Artifacts

With `--artifact-dir`, every shard leaves an audit trail:

```
runs/<run_id>/target_<id>/shard_<n>/shard_input.json
runs/<run_id>/target_<id>/shard_<n>/objective_reports.json
```

Each `ObjectiveReport` carries `evidence` naming the SBR module, the raw metric,
and the value it came from.

## HPC scaling

Scoring is sharded by `(target_id, candidate_batch)` with `shard_size`, and each
shard is an independent subprocess writing its own artifacts — the unit of work
for a Slurm/PBS job array. Shard failure is isolated: `SbrSubprocessUnavailable`
surfaces as `scoring_error` with the worker's stderr rather than a partial
ranking.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

52 tests. The integration tests (`test_sbr_integration.py`, `test_jnana_hook.py`,
`test_inprocess_backend.py`) execute the real StructBioReasoner scoring core and
patch Jnana's real `RankingAgent`; `test_md_engine.py` runs the OpenMM MD
pipeline on a synthetic 2-chain complex built in-process (no network). They skip
automatically when a checkout / OpenMM is absent. Override locations with
`BINDER_ADAPTER_SBR_ROOT` / `BINDER_ADAPTER_JNANA_ROOT`.

## Tier 2 (done) — in-process scoring

`InProcessScoringBackend` is live: select it with `--backend in_process` or
`AdapterConfig(backend="in_process")`. It calls the shared scoring core directly
instead of spawning a subprocess per shard, and a dedicated parity test proves
its reports are identical to the subprocess backend's. `subprocess` remains the
default for process isolation.

## Still dark — the one remaining unavailable objective

`thermostability` and `binding_affinity` are now measurable via the OpenMM MD
path above. Only `structure_confidence` remains unavailable, and it is genuinely
absent tooling on `pydantic_refactor`, not a wiring gap:

- **structure_confidence** — the `TrajectoryAnalysisAgent` path is a dead end and
  must NOT be wired up as-is. Two defects (verified live):
    1. The `ImportError` is not a missing `base_agent.py` — the
       `from ...core.base_agent import BaseAgent` on line 18 is vestigial
       (`class TrajectoryAnalysisAgent:` declares no base). Deleting that one
       dead line fixes the import; restoring `base_agent.py` is unnecessary.
    2. `_calculate_confidence` is defined twice in the class. The real
       RMSD-std/frame-count heuristic (line 128, clamped `[0,1]`) is shadowed by
       a later stub (line 266, `# TODO ...; return 0.75`). Python keeps the last
       def, so both a rich and an empty input return a constant `0.75`.
  Wiring this in unchanged would feed an identical `0.75` to every candidate — an
  objective that looks measured but contributes nothing to Pareto ranking. Real
  options: (a) run `ChaiAgent` on a GPU node for genuine per-candidate
  confidence, or (b) upstream a fix to SBR that removes the shadowing stub — not
  something the adapter should monkey-patch.

For reference, StructBioReasoner's own MM-PBSA path
(`agents/molecular_dynamics/mmpbsa_agent.py`, `FEAgent`) stays unused: it imports
and dispatches (academy + parsl present) but `distributed.parsl_mmpbsa` does
`from molecular_simulations.simulate.mmpbsa import MMPBSA`, and
`molecular_simulations` needs Amber + `numpy>=2` + precomputed trajectories. The
adapter's OpenMM engine sidesteps all of that. (Minor upstream bug to fix if that
path is ever provisioned: `FEAgent.run` references `fe_futures` before defining
it.)
