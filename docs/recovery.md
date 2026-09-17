# Recovering the residential study

The recovery runner preserves every existing result, including failed tasks,
and restarts only executions with no terminal `result.json`. It verifies the
original controller and task hashes. Interrupted records and partial recordings
remain in a dated recovery directory.

```bash
python -m dream_sim.resume_study \
  --run /path/to/live/batch \
  --durable /path/to/durable/batch \
  --gpus 0 1
```

Inspect the printed remaining task list, then add `--execute` to resume. The
deployment allocation is read from `allowed_gpu_ids` in
`.runtime/worker_resource_limits.json`. Use the GPU indices allocated on your machine; the example requires GPUs
0 and 1 in that configuration. Duplicate or
unallocated GPU IDs are rejected. Each worker uses two OpenCV/BLAS threads, up to eight
software rasterizer threads, and two threads per video encoder. The encoder
uses `veryfast` with the original quality, resolution and frame cadence.

The runner keeps at least 128 GiB free in the recording filesystem before
starting another task. It archives one recording at a time, verifies all archive
members and writes progress atomically. A worker error stops further launches;
it does not trigger an automatic retry loop. A process lock prevents two recovery
schedulers from running against the same durable batch.

Batches use disk-backed recording directories. Executing from `/dev/shm` requires the explicit legacy
`--allow-ram-recordings` option. The adapter shares two process locks across
batch schedulers and independent diagnostics; another worker cannot reuse a GPU.
A deployment may configure a memory cgroup, for example 16 GiB per worker
and 32 GiB for two workers together. The portable configuration generator
does not create these kernel limits. The adapter requests a
graceful stop at 90% of the worker limit; the kernel enforces the hard limit.
Each runtime receipt has a companion `.memory.json` with the actual kernel
limit and peak accounted usage. Configuring kernel limits is deployment work;
the portable adapter only applies them when that configuration exists.

`progress.json` contains active task names, physical GPU IDs, process IDs,
completed outcomes, archived counts and errors. The dated recovery directory
contains original records, per-task logs, runtime settings and source hashes.
Physical replay, fold checks and video qualification are separate from task
completion; `release_ready` remains false until those checks are complete.

To run a controller, prepare a **new** complete manifest with the compact
batch launcher's `--prepare-only` option. Run it with `resume_study --execute
--audit-and-prune`: this executes a task once, serially archives it, audits any
task success using its frozen physics/record scripts and the fold checks, then
removes camera files only after their durable archive bytes have been verified.
Results, source snapshots and full raw archives remain available. Failed task
results remain final; a stopped archive/audit resumes without another inference
attempt. A `CANCELLED_BY_USER.json` in either batch directory prevents resumption
and stops further task launches. It does not interrupt a currently running task.

The controller uses a 64 MiB resident RGB-D observation cache backed by saved
frames and a separate 64 MiB OWL image-feature cache. These cache limits are
separate from the memory used by the simulator and models. The encoder uses
CPU rendering, and video exports run serially.
