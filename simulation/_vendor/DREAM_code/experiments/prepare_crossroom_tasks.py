#!/usr/bin/env python3
"""Environment-only multi-room geometry audit and controlled task construction.

Room cores are derived from stage walls (not furniture) and separated by narrow
passages. Figures must be visually checked against the furnished scene. No
geometry raster, room label, endpoint or cart path is supplied to the policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from architecthor_navigation import SCENE_IDS,FETCH_START_XY,_world_stage_mesh,_object_world_vertices,_mesh
from maniskill_replicacad_search_video import NavGrid
from houseexpo_cross_room import weighted_distances,_reconstruct_path


def layers(assets,scene,resolution=.06,config_relative=None,include_objects=True):
    config_file=(assets/f"ai2thor-hab/configs/scenes/ArchitecTHOR/{scene}.scene_instance.json" if config_relative is None
                 else assets/"ai2thor-hab/configs/scenes/ProcTHOR"/config_relative)
    config=json.loads(config_file.read_text())
    stage=assets/"ai2thor-hab/assets"/(config["stage_instance"]["template_name"]+".glb")
    if scene.startswith("ProcTHOR"):
        import trimesh
        mesh=_mesh(stage)
        # Match the installed AI2THORBaseSceneBuilder exactly: Rx(90) Ry(-90).
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[1,0,0]) @
                             trimesh.transformations.rotation_matrix(math.pi/2,[0,-1,0]))
    else:
        mesh=_world_stage_mesh(stage)
    triangles=np.asarray(mesh.triangles)
    normals=np.asarray(mesh.face_normals)
    floor_mask=(normals[:,2]>.75)&(triangles.mean(1)[:,2]>-.10)&(triangles.mean(1)[:,2]<.12)&(triangles[:,:,2].max(1)<.14)
    floors=triangles[floor_mask]
    low=floors[:,:,:2].reshape(-1,2).min(0)-.3
    high=floors[:,:,:2].reshape(-1,2).max(0)+.3
    shape=(np.ceil((high-low)/resolution).astype(int)+1)[::-1]
    floor=np.zeros(shape,np.uint8)
    walls=np.zeros(shape,np.uint8)
    furniture=np.zeros(shape,np.uint8)
    def pixels(xy):
        return np.rint(np.column_stack(((xy[:,0]-low[0])/resolution,(high[1]-xy[:,1])/resolution))).astype(np.int32)
    for triangle in floors:
        cv2.fillConvexPoly(floor,pixels(triangle[:,:2]),1)
    wall_mask=(normals[:,2]<.75)&(triangles[:,:,2].max(1)>.10)&(triangles[:,:,2].min(1)<1.65)
    for triangle in triangles[wall_mask]:
        cv2.fillConvexPoly(walls,pixels(triangle[:,:2]),1)
    for obj in (config["object_instances"] if include_objects else []):
        asset=assets/"ai2thorhab-uncompressed/assets"/(obj["template_name"]+".glb")
        vertices=_object_world_vertices(asset,obj)
        if vertices[:,2].max()<=.10 or vertices[:,2].min()>=1.65:
            continue
        if "doorway" in obj["template_name"].lower():
            # Native doorway assets contain both jambs and an overhead lintel.
            # A single projected convex hull falsely seals an already open
            # passage. Project only actual low geometry, without modifying
            # or removing any environment mesh/collision.
            faces=np.asarray(_mesh(asset).faces)
            triangles=vertices[faces]
            for triangle in triangles[(triangles[:,:,2].max(1)>.10)&(triangles[:,:,2].min(1)<1.65)]:
                cv2.fillConvexPoly(furniture,pixels(triangle[:,:2]),1)
            continue
        hull=cv2.convexHull(pixels(vertices[:,:2]))
        if len(hull)>=3:
            cv2.fillConvexPoly(furniture,hull,1)
    raw=(floor>0)&(walls==0)&(furniture==0)
    stage=(floor>0)&(walls==0)
    grid=NavGrid(raw,low,high,resolution,0)
    return grid,stage,walls,furniture


def room_partition(stage,resolution):
    clearance=distance_transform_edt(stage)*resolution
    # Doorways below 1.2 m separate wall-bounded room cores; unrelated narrow
    # furniture aisles cannot create rooms because furniture is not used here.
    core_mask=stage&(clearance>=.60)
    count,components,stats,_=cv2.connectedComponentsWithStats(core_mask.astype(np.uint8),8)
    cores=np.zeros_like(components,np.int16)
    sizes=[]
    for i in range(1,count):
        area=stats[i,cv2.CC_STAT_AREA]*resolution**2
        if area>=1.5:
            cores[components==i]=len(sizes)+1
            sizes.append(area)
    # Multi-source propagation respects walls. Labels alone are not a new
    # dataset-provided semantic-room annotation; their provenance is explicit.
    labels=cores.copy()
    distances=np.full(stage.shape,np.inf)
    queue=[]
    for row,col in np.argwhere(cores):
        distances[row,col]=0.
        heapq.heappush(queue,(0.,int(row),int(col)))
    while queue:
        dist,row,col=heapq.heappop(queue)
        if dist>distances[row,col]:
            continue
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc=row+dr,col+dc
            if rr<0 or cc<0 or rr>=stage.shape[0] or cc>=stage.shape[1] or not stage[rr,cc]:
                continue
            if dist+1<distances[rr,cc]:
                distances[rr,cc]=dist+1
                labels[rr,cc]=labels[row,col]
                heapq.heappush(queue,(dist+1,rr,cc))
    return cores,labels,sizes


def segment(grid,a,b):
    cells=np.asarray([grid.world_to_cell(p) for p in np.linspace(a,b,max(2,int(np.linalg.norm(np.asarray(a)-b)/.025)+1))])
    if (cells<0).any() or (cells>=np.array(grid.free.shape)).any():
        return False
    return bool(grid.free[cells[:,0],cells[:,1]].all())


def construct(grid,cores,labels,index,variant):
    clearance=distance_transform_edt(grid.free)*grid.resolution_m
    safe=grid.free&(clearance>=.46)
    cells=np.argwhere(safe)
    if not len(cells):
        raise RuntimeError("No conservative Fetch free space")
    xy=np.array([grid.cell_to_world(tuple(c)) for c in cells])
    rooms=labels[cells[:,0],cells[:,1]]
    original=np.asarray(FETCH_START_XY[index])
    home_ids=np.flatnonzero((clearance[cells[:,0],cells[:,1]]>=.63)&(cores[cells[:,0],cells[:,1]]>0))
    home_ids=sorted(home_ids,key=lambda i:np.linalg.norm(xy[i]-original))
    # Geometry-only development alternatives; no learned success is consulted.
    home_ids=home_ids[variant::3][:100]
    for hi in home_ids:
        home=xy[hi]
        radii=np.linalg.norm(xy-home,axis=1)
        initial_ids=np.flatnonzero((rooms==rooms[hi])&(radii>=1.25)&(radii<=1.60))
        initial_ids=sorted(initial_ids,key=lambda i:abs(radii[i]-1.40))[::3][:35]
        bins=np.flatnonzero((rooms==rooms[hi])&(radii>=.68)&(radii<=.74))
        if not len(bins):
            continue
        for ti in initial_ids:
            initial=xy[ti]
            if not segment(grid,home,initial):
                continue
            bi=next((i for i in sorted(bins,key=lambda i:np.dot(xy[i]-home,initial-home)) if segment(grid,home,xy[i])),None)
            if bi is None:
                continue
            bin_xy=xy[bi]
            cart_safe=safe.copy()
            forbidden=(radii<.85)|(np.linalg.norm(xy-bin_xy,axis=1)<.62)
            cart_safe[cells[forbidden,0],cells[forbidden,1]]=False
            if not cart_safe[tuple(cells[ti])]:
                continue
            distances,parents=weighted_distances(cart_safe,tuple(cells[ti]))
            geodesic=distances[cells[:,0],cells[:,1]]*grid.resolution_m
            different=(rooms!=rooms[hi])&(rooms>0)&(cores[cells[:,0],cells[:,1]]>0)
            candidates=np.flatnonzero(different&(geodesic>=4.5)&(geodesic<=12.)&(radii>=3.5)&
                                     (clearance[cells[:,0],cells[:,1]]>=.70))
            if not len(candidates):
                continue
            candidates=sorted(candidates,key=lambda i:abs(geodesic[i]-(6.5+variant)))
            for ei in candidates[:25]:
                path=_reconstruct_path(parents,tuple(cells[ti]),tuple(cells[ei]))
                path_xy=np.array([grid.cell_to_world(tuple(c)) for c in path])
                room_trace=[int(labels[tuple(c)]) for c in path]
                changes=[k for k in range(1,len(path)) if room_trace[k]!=room_trace[k-1]]
                if not changes:
                    continue
                gate_points=[path_xy[k].tolist() for k in changes]
                support_radius=.15
                if not all(clearance[tuple(c)]>=support_radius+.08 for c in path):
                    continue
                return dict(scene=SCENE_IDS[index],seed=117+index+variant*20,spawn_xy=home.tolist(),
                    target_xy=initial.tolist(),endpoint=xy[ei].tolist(),bin_xy=bin_xy.tolist(),
                    disturbance_waypoints=path_xy[4::4].tolist(),disturbance_step=700,
                    query_step=700+int(math.ceil(geodesic[ei]/.05*20))+700,
                    planned_relocation_path_m=float(geodesic[ei]),initial_room=int(rooms[hi]),
                    destination_room=int(rooms[ei]),room_boundary_route_points=gate_points,
                    room_map_file=f"{SCENE_IDS[index]}_environment_rooms.npz",
                    minimum_endpoint_geometry_clearance_m=float(clearance[tuple(cells[ei])]),
                    task_variant=variant,
                    construction="Environment-only wall-bounded room partition and geometry clearance; must be visually audited. No room/endpoint/raster input to policy.")
    raise RuntimeError("No inter-room fixture satisfies geometry rules")


def construct_remembered_room(grid,cores,labels,index,variant,scene_id=None,preferred_spawn=None,whole_house=False):
    """Robot in A recalls a target seen through a doorway in B; target moves
    within B before the request. The robot must enter B and return to A.
    This is a dynamic cross-room retrieval task, not a claim that every object
    relocation itself crosses a room boundary.
    """
    clearance=distance_transform_edt(grid.free)*grid.resolution_m
    safe=grid.free&(clearance>=.42)
    _,components=cv2.connectedComponents(safe.astype(np.uint8),8)
    base_cells=np.argwhere(safe)
    base_xy=np.array([grid.cell_to_world(tuple(c)) for c in base_cells])
    target_cells=np.argwhere(grid.free&(clearance>=.26)&(cores>0))
    target_xy=np.array([grid.cell_to_world(tuple(c)) for c in target_cells])
    target_rooms=labels[target_cells[:,0],target_cells[:,1]]
    base_rooms=labels[base_cells[:,0],base_cells[:,1]]
    original=np.asarray(FETCH_START_XY[index] if preferred_spawn is None else preferred_spawn)
    resolved_scene=SCENE_IDS[index] if scene_id is None else scene_id
    home_ids=np.flatnonzero((clearance[base_cells[:,0],base_cells[:,1]]>=.53)&(cores[base_cells[:,0],base_cells[:,1]]>0))
    if whole_house:
        # Coverage of a larger house is based on geometry, not on any policy
        # outcome. Keep a spatially spaced set instead of only the old spawn.
        home_ids=[i for i in home_ids if (base_cells[i,0]-variant)%4==0 and base_cells[i,1]%4==0]
        home_ids=sorted(home_ids,key=lambda i:np.linalg.norm(base_xy[i]-original))
    else:
        home_ids=sorted(home_ids,key=lambda i:np.linalg.norm(base_xy[i]-original))[variant::4][:180]
    if not len(target_cells):
        raise RuntimeError("No target room cores")
    target_order=sorted(set(target_rooms)-{0})
    target_order=target_order[variant%len(target_order):]+target_order[:variant%len(target_order)]
    for desired_room in target_order:
        for hi in home_ids:
            home=base_xy[hi]
            source_room=int(base_rooms[hi])
            if source_room==desired_room:
                continue
            connected=components[base_cells[:,0],base_cells[:,1]]==components[tuple(base_cells[hi])]
            if not np.any(connected&(base_rooms==desired_room)):
                continue
            radial=np.linalg.norm(target_xy-home,axis=1)
            initial_ids=np.flatnonzero((target_rooms==desired_room)&(radial>=1.25)&(radial<=2.50))
            initial_ids=sorted(initial_ids,key=lambda i:abs(radial[i]-1.8))[::3][:28]
            if not initial_ids:
                continue
            home_distance=np.linalg.norm(base_xy-home,axis=1)
            bins=np.flatnonzero((base_rooms==source_room)&(home_distance>=.68)&(home_distance<=.76))
            if not len(bins):
                continue
            for ti in initial_ids:
                initial=target_xy[ti]
                if not segment(grid,home,initial):
                    continue
                bi=next((i for i in sorted(bins,key=lambda i:np.dot(base_xy[i]-home,initial-home)) if segment(grid,home,base_xy[i])),None)
                if bi is None:
                    continue
                bin_xy=base_xy[bi]
                cart_safe=grid.free&(clearance>=.22)
                rr,cc=np.indices(cart_safe.shape)
                all_xy=np.stack((grid.minimum_xy[0]+cc*grid.resolution_m,grid.maximum_xy[1]-rr*grid.resolution_m),axis=-1)
                cart_safe &= (np.linalg.norm(all_xy-home,axis=-1)>=.72)&(np.linalg.norm(all_xy-bin_xy,axis=-1)>=.55)
                distances,parents=weighted_distances(cart_safe,tuple(target_cells[ti]))
                geodesic=distances[target_cells[:,0],target_cells[:,1]]*grid.resolution_m
                candidates=np.flatnonzero((target_rooms==desired_room)&(geodesic>=1.5)&(geodesic<=6.)&(radial>=3.0))
                candidates=sorted(candidates,key=lambda i:abs(geodesic[i]-(3.0+variant*.5)))[::3][:80]
                dock_cells=base_cells[connected&(base_rooms==desired_room)]
                dock_xy=base_xy[connected&(base_rooms==desired_room)]
                for ei in candidates:
                    endpoint=target_xy[ei]
                    if segment(grid,home,endpoint):
                        continue
                    dock_radius=np.linalg.norm(dock_xy-endpoint,axis=1)
                    docking=(dock_radius>=.58)&(dock_radius<=.66)
                    if not docking.any():
                        continue
                    path=_reconstruct_path(parents,tuple(target_cells[ti]),tuple(target_cells[ei]))
                    if any(labels[tuple(cell)]!=desired_room for cell in path):
                        continue
                    path_xy=np.array([grid.cell_to_world(tuple(c)) for c in path])
                    return dict(scene=resolved_scene,seed=217+index+variant*20,spawn_xy=home.tolist(),
                        target_xy=initial.tolist(),endpoint=endpoint.tolist(),bin_xy=bin_xy.tolist(),
                        disturbance_waypoints=path_xy[4::4].tolist(),disturbance_step=700,
                        query_step=700+int(math.ceil(geodesic[ei]/.05*20))+700,
                        planned_relocation_path_m=float(geodesic[ei]),initial_room=source_room,
                        initial_target_room=int(desired_room),destination_room=int(desired_room),
                        room_boundary_route_points=[],room_map_file=f"{resolved_scene}_environment_rooms.npz",
                        minimum_endpoint_geometry_clearance_m=float(clearance[tuple(target_cells[ei])]),
                        available_geometry_docks=int(docking.sum()),task_variant=variant,
                        construction="Robot in room A observes initial target in B through an open doorway. Target relocates within B while robot is idle; no relocated endpoint or room map is supplied to the policy. Full task requires physical search/grasp in B and return/place in A.")
    raise RuntimeError("No remembered-other-room fixture satisfies geometry rules")


def worker(args):
    assets,output,index,mode,reuse=args
    scene=SCENE_IDS[index]
    cache=None if reuse is None else reuse/f"{scene}_environment_rooms.npz"
    if cache is not None and cache.is_file():
        with np.load(cache) as data:
            grid=NavGrid(data["raw_free"],data["minimum_xy"],data["maximum_xy"],float(data["resolution"]),0)
            stage=data["stage_free"]
            walls=(~stage).astype(np.uint8)
            furniture=(stage&~grid.free).astype(np.uint8)
    else:
        grid,stage,walls,furniture=layers(assets,scene)
    cores,labels,sizes=room_partition(stage,grid.resolution_m)
    np.savez_compressed(output/f"{scene}_environment_rooms.npz",raw_free=grid.free,stage_free=stage,
        room_cores=cores,room_labels=labels,minimum_xy=grid.minimum_xy,maximum_xy=grid.maximum_xy,
        resolution=grid.resolution_m)
    rng=np.random.default_rng(42)
    colors=rng.integers(140,235,size=(max(2,len(sizes)+1),3),dtype=np.uint8)
    colors[0]=(245,245,245)
    canvas=colors[labels]
    canvas[walls.astype(bool)]=(30,40,45)
    canvas[furniture.astype(bool)]=(95,105,115)
    rows=[]
    for variant in range(3):
        try:
            task=(construct_remembered_room if mode=="remembered-room" else construct)(grid,cores,labels,index,variant)
            name=f"{index+1:02d}_{scene}_layout{variant+1:02d}.json"
            (output/name).write_text(json.dumps(task,indent=2)+"\n")
            rows.append(dict(status="constructed_not_executed",file=name,**task))
            if variant==0:
                pts=np.array([grid.world_to_cell(p)[::-1] for p in [task["target_xy"],*task["disturbance_waypoints"],task["endpoint"]]],np.int32)
                cv2.polylines(canvas,[pts],False,(20,40,210),2)
                for field,color in (("spawn_xy",(230,70,20)),("target_xy",(0,200,240)),("endpoint",(20,20,230)),("bin_xy",(20,150,20))):
                    cv2.circle(canvas,grid.world_to_cell(task[field])[::-1],4,color,-1)
        except Exception as error:
            rows.append(dict(status="geometry_configuration_failed",scene=scene,variant=variant,error=repr(error)))
    for label in range(1,len(sizes)+1):
        cells=np.argwhere(cores==label)
        center=tuple(np.mean(cells,axis=0).astype(int)[::-1])
        cv2.putText(canvas,f"R{label}",center,cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(output/f"{scene}_environment_rooms.png"),cv2.resize(canvas,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
    report=dict(scene=scene,room_core_count=len(sizes),room_core_areas_m2=sizes,tasks=rows)
    (output/f"{scene}_construction.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--mode",choices=("relocation-crosses-room","remembered-room"),default="relocation-crosses-room")
    parser.add_argument("--reuse-layers",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(worker,[(args.assets,args.output,i,args.mode,args.reuse_layers) for i in range(len(SCENE_IDS))]))
    (args.output/"construction_report.json").write_text(json.dumps(reports,indent=2)+"\n")
    print(json.dumps(reports,indent=2),flush=True)


if __name__=="__main__":
    main()
