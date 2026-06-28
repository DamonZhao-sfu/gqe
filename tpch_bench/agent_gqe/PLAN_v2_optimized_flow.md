# Plan v2: optimizing our GPU-codegen agent flow (lessons from BespokeOLAP & GenDB)

Status: **PLAN ONLY.** Research of two external systems + a concrete proposal to upgrade our flow.
Sources:
- BespokeOLAP — https://github.com/DataManagementLab/BespokeOLAP
- GenDB — https://github.com/SolidLao/GenDB

## 1. What they do (distilled)

**BespokeOLAP** — LLM synthesizes a custom C++ OLAP engine per workload.
- Agentic loop with tools: ApplyPatch / Shell / compile / run.
- 3 phases: (1) storage-plan generation, (2) base impl (loader/builder/query), (3) **optimization
  loop** guided by speedup (continue / revert / terminate).
- Correctness via **DuckDB at multiple scale factors** (QueryValidator).
- **Hot-reload**: loader/builder/query compiled as shared libs, reloaded without restart.
- JSON-persisted conversations for exact replay. Eval: TPC-H + CEB.

**GenDB** — 5 collaborating agents emit instance-specific native C++.
- Agents: Workload Analyzer (profiles cache/cores/SIMD + join/group cardinalities) → Storage/Index
  Designer → Query Planner (resource-aware) → Code Generator → **Query Optimizer** (runtime-feedback
  iteration; e.g. Q18 12s→74ms).
- Hardware-aware codegen (L1 direct accumulators vs L3-aware + lock-free CAS by group cardinality).
- Eval: TPC-H + **unseen** SEC-EDGAR (proves instance-level optimization, not memorization).

## 2. Common winning patterns to adopt
1. **A real optimization loop** keyed on a *speedup* reward (not just correctness): keep-if-faster,
   revert-on-regression, stop-on-plateau.
2. **DuckDB oracle at multiple scale factors** (small SF for the fast correctness loop, larger SF for
   the perf loop). We already use DuckDB; add the multi-SF discipline.
3. **Phase decomposition** (analyze → plan → generate → verify → optimize) instead of one shot.
4. **Profile-as-context**: feed the agent the workload profile (table sizes, filter selectivity, join
   cardinalities) and the GPU profile (arch, SM count, memory) so it makes *instance-level* choices.
5. **Persisted conversations / results** (JSON) for replay + building a fine-tuning/RL dataset.
6. **Generalization eval** on a non-TPC workload to show it isn't memorizing.

## 3. Gaps in our current flow
We have two correctness loops (`tpch_bench/agent/` libcudf; `tpch_bench/agent_gqe/` gqe DSL): generate
→ compile → run → diff vs DuckDB → retry. Missing: the **optimization loop**, **profiling context**,
**multi-SF**, **persistence**, and **generalization eval**.

## 4. Proposed pipeline (adapted to our GPU / GQE setting)

```
A. Analyze ─▶ B. Plan ─▶ C. Generate ─▶ D. Verify ─▶ E. Optimize ─▶ F. Persist
   (profile)    (strategy)  (DSL code)    (correct?)   (faster?)      (replay/dataset)
                                  ▲___________|              |
                                  └── fix compile/run/diff ──┘ (correctness loop, have)
                                                              ▲___________|
                                                              └ keep/revert by speedup (NEW)
```

- **A. Analyze (new).** A profiling step (DuckDB + GPU query) produces a JSON workload profile:
  per-table row counts, the query's predicate selectivities, join build/probe sizes, group-by
  cardinalities; plus GPU arch/SM/memory. This is the "instance" context GenDB relies on.
- **B. Plan (new, light).** A planner prompt turns SQL + profile into a strategy note: join order,
  which dimensions are selective (probe order), whether a fused UDR kernel is worth it, which gqe
  knobs to try. Output is short structured text the codegen step consumes.
- **C. Generate.** Emit `build_plan_gen.cpp` with our name-based DSL (have). The strategy from B
  guides join order / column pruning.
- **D. Verify (have, extend).** Compile (incremental ninja) → run → diff vs DuckDB. Do it at
  **SF=0.01 first** (fast) then **SF=1** to catch scale-dependent bugs. Loop on errors.
- **E. Optimize (new — the core upgrade).** Once correct, iterate for speed with a *speedup reward*:
  - Tune gqe knobs without regenerating the plan: `GQE_JOIN_USE_PERFECT_HASH`,
    `GQE_AGGREGATION_USE_PERFECT_HASH`, `storage_kind=device_memory`, `MAX_NUM_PARTITIONS`,
    `GQE_NUM_ROW_GROUPS`. Measure with `compare`/`profile_udr` (nsys kernel time). Keep if faster,
    revert if slower; stop after K no-improvement rounds.
  - Escalate to a **fused UDR kernel** for the hot join (generate `qN_udr.cu`) using
    `q3_udr.cu`/`q7_udr.cu`/`q43_udr.cu` as few-shot; reward = nsys kernel time. This is our analog
    of GenDB's instance-level code specialization, and where the engine-IR path pays off.
- **F. Persist (new).** Append every (prompt, response, build log, run result, timing) to JSONL per
  query/run for exact replay (BespokeOLAP-style) and as a dataset for later fine-tuning/RL.

## 5. Concrete changes to our repo (when we implement)
1. `agent_gqe/profile.py` — emit workload+GPU profile JSON (DuckDB counts/selectivity + `nvidia-smi`/
   device query). Feed into the prompt.
2. `agent_gqe/optimize.py` (or extend `gqe_codegen.py`) — Phase E knob-tuning loop with
   keep/revert/plateau on a timing reward (reuse `compare_udr.sh`/`profile_udr.sh`).
3. Multi-SF: generate references + verify at SF 0.01 and 1; perf-measure at SF ≥ 1.
4. `--log-jsonl` in both agents: persist conversations + metrics (replay + dataset).
5. Reward = (correct ? 1 : 0) gate, then minimize runtime (engine ms; nsys kernel ms for UDR).
   Leaderboard CSV per query (best plan/knobs/kernel + time).
6. (Later) UDR-escalation agent: prompt the model to emit a `qN_udr.cu` fused kernel when Phase E
   knob-tuning plateaus; few-shot with the existing `*_udr.cu`.
7. (Later) Generalization eval: a small non-TPC workload (our SEC-EDGAR analog) to confirm gains
   aren't memorized.

## 6. What we keep vs change
- **Keep**: name-based DSL (lower error rate than raw index code), DuckDB oracle, fixed-scaffold
  "fill build_plan" (smaller action space than free-form file editing), incremental ninja build.
- **Change/add**: the optimization loop + speedup reward (biggest win), profiling context, multi-SF,
  JSONL persistence, and the UDR-kernel escalation path.
- **Deliberately NOT copying**: full free-form agent-with-shell-tools autonomy (BespokeOLAP) — our
  constrained loop is more reproducible and cheaper; we adopt their *optimization loop* idea, not the
  unconstrained editing. We can revisit giving the model tools if the constrained loop stalls.

## 7. Phased rollout
- **P1**: add Phase E knob-tuning + speedup reward + leaderboard on the 3 working queries (q3/q7/q43).
- **P2**: add profiling context (A/B) and multi-SF verify; measure if it improves convergence/quality.
- **P3**: UDR-kernel escalation (generate `qN_udr.cu`) + JSONL persistence.
- **P4**: generalization eval on a non-TPC workload.

## 8. Open questions for the user
1. Optimization reward: end-to-end engine time, or nsys **kernel** time (I/O-excluded)? (kernel time
   shows the real win but needs nsys.)
2. Knob-tuning only, or also auto-escalate to generating UDR fused kernels in Phase E?
3. Single-agent (one model, phased prompts) or true multi-agent (separate planner/codegen/optimizer
   like GenDB)? Single-agent is simpler; multi-agent matches GenDB more closely.
4. Add JSONL conversation/metric logging now (useful for a future RL/fine-tuning dataset)?
