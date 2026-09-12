"""Check task initialization and both camera views without running a policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from dream_sim.sources import PROJECT, engine_root, safe_member
from dream_sim.study import controller_overrides, load_task_manifest, materialize_controller


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def initialize_worker(experiments, asset_dir):
    os.environ.update(MS_ASSET_DIR=asset_dir, CUDA_VISIBLE_DEVICES='-1', LP_NUM_THREADS='4',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    os.environ.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/lvp_icd.json')
    sys.path.insert(0, experiments)


def check_task(index, task_path, asset_dir, image_dir):
    import gc
    import numpy as np
    import torch
    from maniskill_learned_probe import make_env, initialize_compact_arm, SimulatorIO
    from instruction_environment import sample_start, create_fixture, exclude_initial_categories
    from mani_skill.utils.structs.pose import Pose

    started=time.time(); env=None
    task=json.loads(Path(task_path).read_text())
    result=dict(index=index, scene=task['scene'], seed=task['seed'], initialized=False,
                rendered=False, policy_executed=False, control_steps=0)
    torch.set_num_threads(2)
    try:
        with np.load(Path(task_path).parent / task['room_map_file']) as data:
            room={key:data[key] for key in data.files}
        env=make_env(task['scene'], width=640, height=480)
        env.reset(seed=task['seed']); base=env.unwrapped
        spawn,yaw,_=sample_start(room,task['initial_room'],task['seed'],
            [(task['placement_table_xy'],task['placement_table_collision_radius_m'])],
            target_xy=task['target_xy'],docking_xy=task['endpoint'])
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*spawn,.02]]))
        q=base.agent.robot.get_qpos().clone(); q[:,2]=yaw
        base.agent.robot.set_qpos(q); base.agent.controller.reset(); initialize_compact_arm(base)
        categories={task['recipe']['environment_assets'][role].rsplit('_',1)[0]
                    for role in ('pickup','placement')}
        exclude_initial_categories(base,categories)
        create_fixture(base,Path(asset_dir)/'data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects',task)
        result['initialized']=True
        io=SimulatorIO(env); observation=io.capture(); overview=io.overview()
        if observation.rgb.shape!=(480,640,3) or observation.depth_m.shape!=(480,640):
            raise ValueError('Unexpected RGB-D dimensions')
        if not np.isfinite(observation.depth_m).all() or overview.ndim!=3 or overview.shape[2]!=3:
            raise ValueError('Invalid camera output')
        if image_dir:
            import cv2
            path=Path(image_dir)/f'{index:02d}.jpg'
            if not cv2.imwrite(str(path),cv2.cvtColor(observation.rgb,cv2.COLOR_RGB2BGR)):
                raise OSError('Could not save camera preview')
            result['preview']=f'previews/{index:02d}.jpg'
        result.update(rendered=True,rgb_shape=list(observation.rgb.shape),
                      depth_shape=list(observation.depth_m.shape),overview_shape=list(overview.shape))
    except Exception as error:
        result['error']=repr(error)
    finally:
        if env is not None: env.close()
        gc.collect()
    result['wall_time_s']=time.time()-started
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',choices=('recovery_v3','recovery_v4','recovery_v5','recovery_v6'),default='recovery_v3')
    parser.add_argument('--task-manifest',type=Path,default=PROJECT/'configs/residential50/task_manifest.json')
    parser.add_argument('--asset-lock',type=Path,default=PROJECT/'configs/residential50/render_assets.lock.json')
    parser.add_argument('--asset-dir',type=Path,default=PROJECT/'.runtime/render_assets')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--save-previews',action='store_true')
    args=parser.parse_args()
    if args.workers<1: parser.error('--workers must be positive')
    manifest,tasks=load_task_manifest(args.task_manifest)
    asset_dir=args.asset_dir.resolve(); output=args.output.resolve()
    if output.exists(): raise FileExistsError('Choose a new output directory')
    lock=json.loads(args.asset_lock.read_text())
    if not lock.get('complete') or lock.get('failures'): raise ValueError('Incomplete asset lock')
    for name,expected in lock['artifacts'].items():
        path=asset_dir/'data/scene_datasets'/safe_member(name)
        if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            raise ValueError(f'Asset differs from lock: {name}')
    engine=engine_root(); sys.path.insert(0,str(engine/'experiments'))
    from run_instruction_profile import load_catalog, verify_source
    root,catalog=load_catalog(engine/'experiments/repro_profiles/profiles.json')
    source_id=catalog['study']['source_id']; source=verify_source(root,catalog['sources'][source_id])
    revision=PROJECT/'controllers'/args.controller
    output.mkdir(parents=True); frozen=output/'frozen_workspace/DREAM_code'
    materialize_controller(source,frozen,controller_overrides(revision,source_id))
    image_dir=output/'previews' if args.save_previews else None
    if image_dir: image_dir.mkdir()
    rows=[]
    report=dict(planned=len(tasks),checked=0,all_rendered=False,policy_executed=False,
        task_manifest_sha256=digest(args.task_manifest),asset_lock_sha256=digest(args.asset_lock),
        controller_manifest_sha256=digest(revision/'controller.json'),results=rows)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(str(frozen/'experiments'),str(asset_dir))) as pool:
        futures=[pool.submit(check_task,i,str(task),str(asset_dir),str(image_dir) if image_dir else None)
                 for i,task in enumerate(tasks,1)]
        for future in as_completed(futures):
            row=future.result(); rows.append(row); rows.sort(key=lambda r:r['index'])
            report.update(checked=len(rows),all_rendered=len(rows)==len(tasks) and all(r['rendered'] for r in rows))
            temporary=output/'scene_checks.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(output/'scene_checks.json')
            print(json.dumps(row),flush=True)
    if not report['all_rendered']: raise SystemExit(1)


if __name__=='__main__': main()
