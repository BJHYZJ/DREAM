"""Read-only contact monitoring at every native PhysX substep during replay."""
from collections import defaultdict
import re

import numpy as np


class ContactAudit:
    def __init__(self,base,*,target_name="learned_household_target",fixture_names=None):
        self.base=base
        self.target_name=target_name
        self.fixture_names=set(fixture_names if fixture_names is not None else
                               ("known_delivery_bin","learned_dynamic_support"))
        self.entities={link._objs[0].entity:link.name for link in base.agent.robot.get_links()}
        self.current_step=0
        self.phase=""
        self.samples=0
        self.rows=[]
        self.pairs=defaultdict(lambda:dict(substeps=0,max_force_n=0.))
        self.original=base._after_simulation_step
        base._after_simulation_step=self.after

    def after(self):
        self.original()
        self.samples+=1
        for contact in self.base.scene.get_contacts():
            a,b=(body.entity for body in contact.bodies)
            if a in self.entities and b in self.entities:
                continue
            a_name=re.sub(r"^scene-\d+_","",a.name)
            b_name=re.sub(r"^scene-\d+_","",b.name)
            if a in self.entities or b in self.entities:
                robot,other=(a,b) if a in self.entities else (b,a)
                body_name=self.entities[robot]
                body_kind="robot"
            elif (a_name==self.target_name) != (b_name==self.target_name):
                other=b if a_name==self.target_name else a
                body_name=self.target_name
                body_kind="payload"
            else:
                continue
            name=other.name
            # ManiSkill prefixes the underlying SAPIEN entity name even
            # though the high-level Actor.name is unprefixed.
            canonical=re.sub(r"^scene-\d+_","",name)
            if canonical==self.target_name:
                continue
            force=0.
            for point in contact.points:
                # Exclude support-floor contact, not vertical wall contacts.
                if point.position[2]<.12 and abs(point.normal[2])>.9:
                    continue
                force+=float(np.linalg.norm(point.impulse))/float(self.base.scene.px.timestep)
            if force<.5:
                continue
            kind="task_fixture" if canonical in self.fixture_names else "native_environment"
            key=(self.phase,body_name,name,kind,body_kind)
            self.pairs[key]["substeps"]+=1
            self.pairs[key]["max_force_n"]=max(force,self.pairs[key]["max_force_n"])
            if kind=="native_environment":
                self.rows.append(dict(control_step=self.current_step,physics_substep=self.samples,
                    phase=self.phase,robot_link=body_name,body_kind=body_kind,other=name,force_n=force))

    def report(self):
        return dict(physics_substeps=self.samples,physics_dt_s=float(self.base.scene.px.timestep),
            scope="Read-only contact sampling after every native PhysX substep during exact control re-execution. Includes robot/native and payload/native contacts. Robot self/target contacts and low horizontal floor support are excluded. >=0.5 N contacts retained; fixtures distinguished from native environment.",
            includes_payload_native_contacts=True,
            pairs=[dict(phase=key[0],robot_link=key[1],other=key[2],kind=key[3],body_kind=key[4],**value) for key,value in self.pairs.items()],
            native_environment_contact_rows=self.rows,
            native_environment_contact_control_steps=len({r["control_step"] for r in self.rows}))
