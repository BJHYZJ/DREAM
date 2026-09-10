from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import importlib.util
np=pytest.importorskip('numpy')
pytest.importorskip('scipy')
path=Path(__file__).resolve().parents[1]/'controllers/recovery/experiments/instruction_geometry.py'
spec=importlib.util.spec_from_file_location('dream_recovery_geometry',path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
visible_inner_wall=module.visible_inner_wall


def surface(height):
    y,x=np.mgrid[-.1:.1:101j,-.1:.1:101j]
    xyz=np.stack((x,y,height(x,y)),axis=-1)
    camera=np.eye(4);camera[:3,3]=[0.,0.,1.5]
    obs=SimpleNamespace(world_points=lambda:xyz,camera_to_world_cv=camera)
    box=SimpleNamespace(box_xyxy=[0,0,101,101])
    return obs,box


def test_concave_inner_wall_supplies_observed_cavity_evidence():
    obs,box=surface(lambda x,y:.78+8*(x*x+y*y))
    points=visible_inner_wall(obs,box,[0,0],.07,.90,.75)
    assert len(points)>=12
    assert points[:,2].max()<.88


@pytest.mark.parametrize('height',[
    lambda x,y:np.full_like(x,.8),
    lambda x,y:.87-8*(x*x+y*y),
    lambda x,y:np.where(x<0,.78,.90),
    lambda x,y:np.full_like(x,np.nan),
])
def test_flat_solid_discontinuous_or_missing_surface_cannot_certify_cavity(height):
    obs,box=surface(height)
    assert len(visible_inner_wall(obs,box,[0,0],.07,.90,.75))==0
