"""Observed-depth Fetch adapter around the repository's actual AStar.

No simulator actor, mesh, target endpoint or scene raster enters this module.
Navigation occupancy keeps 3-D occupied evidence so a floor ray cannot erase
a table/wall at another height. Geometric calibration is embodiment-specific.
"""
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from dream_learned_core import ObservedOccupancy
from dream.motion.algo.a_star import AStar


class FetchObservedMap(ObservedOccupancy):
    def __init__(self, origin=(-16., -16.), size=400, resolution=.08, radius=.42):
        super().__init__(origin, size, resolution, radius)
        self.occupied_xyz = np.empty((0, 3), np.float32)
        self.inferred_blocked = np.zeros_like(self.known, bool)
        self.update_count = 0
        self._clearance = None

    def integrate(self, obs, stride=6, self_spheres=()):
        camera = obs.camera_to_world_cv[:3, 3]
        xyz = obs.world_points()[::stride, ::stride].reshape(-1, 3)
        depth = obs.depth_m[::stride, ::stride].reshape(-1)
        valid=np.isfinite(xyz).all(axis=1) & np.isfinite(depth) & (depth>=.10)
        xyz,depth=xyz[valid],depth[valid]
        # A valid distant surface certifies the first four metres of its ray
        # as free; it must not become an obstacle at the clipping plane.
        ray_end=camera+(xyz-camera)*np.minimum(1.,4./depth)[:,None]
        if len(self.occupied_xyz):
            old = self.occupied_xyz
            local = (old-camera) @ obs.camera_to_world_cv[:3, :3]
            projected = local @ obs.intrinsics.T
            pixels = np.rint(projected[:, :2]/np.maximum(projected[:, 2:3], 1e-6)).astype(int)
            inside = (local[:, 2] > .1) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
            inside &= (pixels[:, 0] < obs.depth_m.shape[1]) & (pixels[:, 1] < obs.depth_m.shape[0])
            ids = np.flatnonzero(inside)
            measured = obs.depth_m[pixels[ids, 1], pixels[ids, 0]]
            # Clear only occupied 3-D cells for which the live calibrated ray
            # actually continues beyond that location. Occluded cells remain.
            clear = ids[(measured > local[ids, 2]+.12) & (local[ids, 2]<=4.) & np.isfinite(measured)]
            keep = np.ones(len(old), bool)
            keep[clear] = False
            self.occupied_xyz = old[keep]
        free = np.zeros_like(self.known, np.uint8)
        if len(xyz):
            # Vectorized depth rays; column free evidence cannot overwrite the
            # occupied 3-D store used below.
            ray_samples=max(52,int(np.ceil(4./self.resolution))+1)
            rays = camera + np.linspace(0, 1, ray_samples)[:, None, None]*(ray_end-camera)
            slab = (rays[:, :, 2] >= .10) & (rays[:, :, 2] <= 1.65)
            cells = self.cells(rays[:, :, :2][slab])
            cells = cells[self.inside(cells)]
            free[cells[:, 0], cells[:, 1]] = 1
            hit = (xyz[:, 2] >= .12) & (xyz[:, 2] <= 1.65) & (depth<=4.)
            hit &= np.linalg.norm(xyz[:, :2]-obs.base_xyyaw[:2], axis=1) > .34
            for center, radius in self_spheres:
                hit &= np.linalg.norm(xyz-np.asarray(center), axis=1) > radius
            occupied = np.vstack((self.occupied_xyz, xyz[hit]))
            if len(occupied):
                _, ids = np.unique(np.floor(occupied/self.resolution).astype(np.int32), axis=0, return_index=True)
                self.occupied_xyz = occupied[ids].astype(np.float32)
        # Current footprint is proprioceptive evidence, not a revealed map.
        base = self.cells(obs.base_xyyaw[:2])
        cv2.circle(free, tuple(base[::-1]), max(1, int(.33/self.resolution)), 1, -1)
        self.known[self.known == -1] = 1
        self.known[free.astype(bool)] = 1
        blocked = self.cells(self.occupied_xyz[:, :2])
        blocked = blocked[self.inside(blocked)]
        self.known[blocked[:, 0], blocked[:, 1]] = -1
        self.last_seen[free.astype(bool)] = obs.frame_id
        self.last_seen[blocked[:, 0], blocked[:, 1]] = obs.frame_id
        self.update_count += 1
        self._clearance = None

    def clearance(self):
        if self._clearance is None:
            blocked = (self.known == -1) | self.inferred_blocked
            self._clearance = distance_transform_edt(~blocked)*self.resolution
        return self._clearance

    def traversable(self, radius=None):
        return (self.known == 1) & (self.clearance() >= (self.radius if radius is None else radius))

    def segment_safe(self, start, end, radius=None):
        length = np.linalg.norm(np.asarray(end)-start)
        cells = self.cells(np.linspace(start, end, max(2, int(length/(self.resolution*.3))+1)))
        if not self.inside(cells).all():
            return False
        return bool(self.traversable(radius)[cells[:, 0], cells[:, 1]].all())

    def infer_stall_obstacle(self, base_xy, heading, distance=.60):
        # Contact/stagnation inference: obstacle in the attempted direction,
        # never the privileged coordinate of the collided simulator actor.
        if not np.isfinite(distance) or distance<=0:
            raise ValueError("Stall inference needs a positive finite distance")
        point = np.asarray(base_xy)+distance*np.array([np.cos(heading), np.sin(heading)])
        cell = self.cells(point)
        if self.inside(cell):
            cv2.circle(self.inferred_blocked.view(np.uint8), tuple(cell[::-1]), 1, 1, -1)
            self._clearance = None


class FetchAStar(AStar):
    """Use AStar.run_astar/clean_path with Fetch-observed occupancy.

    The distance-to-obstacle implementation accelerates the original local
    obstacle punishment. Additional clearance is a soft path cost, not an
    optimality claim or access to the unobserved environment.
    """
    def __init__(self, occupancy):
        self.occupancy = occupancy
        space = SimpleNamespace(
            voxel_map=SimpleNamespace(get_2d_map=self._map),
            to_pt=lambda xy: tuple(occupancy.cells(xy)),
            to_xy=lambda pt: tuple(occupancy.world(pt)),
            is_valid=lambda xy: bool(occupancy.segment_safe(xy[:2], xy[:2])),
        )
        super().__init__(space)

    def _map(self):
        safe = self.occupancy.traversable()
        return ~safe, self.occupancy.known != 0

    def reset(self):
        super().reset()
        self.obstacle_distance = distance_transform_edt(self._navigable)

    def compute_obstacle_punishment(self, a, weight, avoid):
        if self.point_is_occupied(*a):
            return float(weight)
        distance = self.obstacle_distance[tuple(a)]
        return float(weight/max(1., distance)) if distance <= avoid else 0.

    def path(self, start_xy, goal_xy):
        self.reset()
        if not self.occupancy.segment_safe(start_xy, start_xy) or not self.occupancy.segment_safe(goal_xy, goal_xy):
            return []
        points = self.run_astar(tuple(start_xy), tuple(goal_xy), remove_line_of_sight_points=False)
        return [] if points is None else [tuple(self.occupancy.cells(p)) for p in points]
