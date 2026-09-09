# DREAM: indoor simulation reproduction

This is the **simulation** branch of [DREAM](https://github.com/BJHYZJ/DREAM).
The ROS real-robot implementation and hardware setup remain on
[`realtime`](https://github.com/BJHYZJ/DREAM/tree/realtime). This branch needs
neither ROS nor the physical robot's drivers.

Ten selected demonstrations show observation-driven cross-room search, a target
relocated after discovery, memory updates, rediscovery, and automatic pick/place
in ten distinct ManiSkill indoor houses. Watch
[Video 01–10](https://bjhyzj.github.io/dream-web/simulation/).
These are selected qualitative demonstrations, not a held-out benchmark or a
population success-rate estimate.

## Get started

```bash
git clone --branch simulation https://github.com/BJHYZJ/DREAM.git
cd DREAM
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r requirements/requirements-learned.txt
python -m pip install --no-deps -e .
python -m pip check
```

Follow the [reproduction tutorial](docs/reproduction.md) to prepare the locked
models/assets and Vulkan/CUDA runtime. After preparation:

```bash
python -m dream_sim.run --preflight
python -m dream_sim.run --case 01 --output results/my_case01
python -m dream_sim.run --all --output results/my_ten_cases
```

The runner makes **one fresh policy attempt per case**, followed by separate
physics/record checks. It does not play saved actions as a policy, retry failed
cases, change seeds, or overwrite previous attempts. Keep every new outcome.
Recorded outcomes are reproducible in the documented same-server environment;
identical behavior on arbitrary hardware is not guaranteed.

## Repository layout

```text
src/dream_sim/                   public execution, preparation and audit package
configs/                        numbered cases, readable tasks and dependency locks
requirements/                   exact Python runtime package versions
tests/                          offline integrity, runner and regression checks
docs/                           environment, execution and architecture tutorials
reproducibility/source_archives/ byte-preserved historical Python source bundle
reproducibility/evidence/        original records, including failed attempts
```

The ten recordings used six historical controller versions; a seventh belongs
to the separate comparison. They have not been silently replaced by one latest
controller. The [architecture guide](docs/architecture.md) explains the source
archive, its integrity checks and how to inspect the exact policy for each video.
All historical source bytes are retained; archive extraction is automatic.

```bash
python -m dream_sim.sources --case 01
python -m dream_sim.run --preflight --dry-run
python -m dream_sim.verify_evidence
python -m pytest -q
```

These offline checks do not execute experiments. The
[video manifest](reproducibility/evidence/gallery/manifest.json) records full
scene IDs, seeds, source hashes and video checksums. Upstream Train/Val/Test
names do not select a controller training/test mode.

The [implementation boundaries](docs/reproduction.md#implementation-boundary)
and [original reproduction records](reproducibility/evidence/reproduction/README.md)
distinguish observation-grounded geometric grasping, simulator odometry and
bounded instruction parsing from the complete physical DREAM stack.
