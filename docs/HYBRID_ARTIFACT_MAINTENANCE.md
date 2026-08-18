# Hybrid Artifact Maintenance Advanced

`MiniMaxH3HybridArtifactMaintenanceT8Advanced` is an isolated Experimental
housekeeping node for the small, content-addressed Hybrid artifacts created by
this project. It does not accept an arbitrary path. The connected, fully
SHA-256-verified Hybrid plan is used to derive the only artifact and sidecar
that the node may inspect or move.

The default is deliberately inert:

- `action=inspect_only`
- `confirm_action=false`
- `operation_epoch=0`

This mode validates and reports the active artifact pair, matching temporary
files, build/maintenance locks, and an existing transaction. It does not create
the transaction or recycle directories and does not modify timestamps or file
contents.

## Actions

| Action | Effect |
|---|---|
| `inspect_only` | Side-effect-free inspection of the exact plan-derived path. |
| `quarantine_artifact_exp` | Verifies the artifact, sidecar, embedded manifest, identity, size, and SHA-256 before moving the pair into the internal `_recycle` directory. |
| `restore_quarantined_exp` | Restores the complete pair for the same epoch. Occupied active paths are never overwritten. |
| `recover_interrupted_exp` | Reconciles an interrupted move back to active paths using the journal and per-file hashes. |
| `quarantine_stale_build_residue_exp` | Moves only stale residue belonging to the exact plan: an orphan artifact/sidecar, its build lock, and matching build temp prefixes. |

Every mutating action requires both `confirm_action=true` and a positive
`operation_epoch`. Use a new epoch for a new quarantine. Reuse exactly the same
epoch only to replay, restore, or recover that transaction.

Quarantine is a recoverable same-volume move, not permanent deletion or secure
erasure. Files remain under:

```text
ComfyUI/models/h3_hybrid_artifacts/_recycle/
```

Review them before any manual final deletion.

## Transaction and crash contract

Before the first move, the node writes an atomic, fsynced journal below
`_maintenance_transactions`. The journal fixes the action, artifact identity,
epoch, source/recycle paths, byte sizes, SHA-256 values, phase, and moved count.
It is updated after every file move.

The loader rejects:

- unsupported phases or phase/count combinations;
- duplicate, missing, additional, or escaped paths;
- a quarantine journal that is not exactly the artifact plus its sidecar;
- symbolic links, non-regular files, size changes, or SHA-256 changes;
- a noncanonical Hybrid manifest or identity;
- an attempt to reuse an epoch for another action.

The Windows implementation checks process handles and exit codes rather than
relying on `os.kill(pid, 0)`. A real subprocess test kills the worker between
the artifact and sidecar moves, ages the orphaned lock, and verifies that
explicit recovery archives the stale lock and restores a valid pair. If owner
liveness cannot be proved safe, the operation fails closed.

## Stale residue rules

The stale age defaults to 60 minutes and can never be lower than 60 seconds.
A build lock whose PID is proven running is refused. No directory scan for
unrelated artifacts or diffusion checkpoints is performed. After incomplete
build residue is quarantined, the normal Artifact Builder may rebuild the exact
content-addressed artifact.

## What this node does not do

- It does not delete or modify FL2VA/Ref2VA source checkpoints.
- It does not scan arbitrary model directories.
- It does not clear ComfyUI execution cache or unload a `MODEL` already held by
  a graph.
- It does not release VRAM, change VBAR/DynamicVRAM, or provide a
  `memory_safe`/`never_oom` guarantee.
- It does not make the Hybrid recipe a proven quality winner or a true
  reference-only AdaLN route.

Use the separate VRAM Policy Advanced node before Hybrid model loading when a
validated VBAR headroom policy is desired. Artifact maintenance and runtime
memory management are intentionally separate responsibilities.

## Examples

- API prompt: `tests/fixtures/api/hybrid_artifact_maintenance_api.json`
- Frontend workflow: `examples/workflows/2026-08-12_H3_Hybrid_Artifact_Maintenance_Advanced.json`

Both examples ship in the safe inspection configuration. Change an action only
after reading the report, and never reuse an old epoch for a new quarantine.
