"""Display-only reframing of an exact control replay, never an acting policy.

The overview is newly rendered from the actual re-executed physics. The head
observation and memory panels retain the accepted original recording. This is
explicitly an alternate spectator view, not another learned-policy experiment.
No hidden target position or scene map is used even to frame this camera.
"""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from mani_skill.utils import sapien_utils

from maniskill_learned_probe import array,sha
from review_instruction_record import logged_captions


def require_strict_record_review(review):
    """Accept known strict-review schemas, never primary-only qualification."""
    version=review.get("review_definition_version")
    checks=review.get("checks",{})
    if (version not in (2,4) or review.get("record_review_passed") is not True
            or not checks or any(value is not True for value in checks.values())
            or (version==4 and review.get("primary_task_record_review_passed") is not True)):
        raise ValueError("Alternate spectator recording requires a passing strict version-2 or version-4 record audit; primary-only success is insufficient")


def spectator_geometry(base,tcp,yaw,phase,mode="overhead"):
    """Display camera from measured robot state only; never returns policy input."""
    if mode not in ("overhead","tool-front-placement"):
        raise ValueError("Unknown spectator framing mode")
    heading=np.array([np.cos(yaw),np.sin(yaw)])
    side=np.array([-heading[1],heading[0]])
    focus_xy=(base[:2]+tcp[:2])*.5
    eye=np.r_[focus_xy-.12*heading-.12*side,2.15]
    look=np.r_[focus_xy,.72]
    fovy=1.38
    if mode=="tool-front-placement" and phase in ("Place in observed receptacle","Check released placement"):
        offset=tcp[:2]-base[:2]
        direction=offset/np.linalg.norm(offset) if np.linalg.norm(offset)>.05 else heading
        # Look back from in front of the tool during release. The camera is
        # display-only and uses no target/receptacle pose or evaluator geometry.
        eye=np.r_[tcp[:2]+.50*direction,2.15]
        look=np.r_[tcp[:2],max(.65,float(tcp[2])-.10)]
        fovy=1.12
    return eye,look,fovy


class InstructionReplayVideo:
    def __init__(self,source,review_path,output,mode="overhead"):
        self.source,self.output=source,output
        if mode not in ("overhead","tool-front-placement"):
            raise ValueError("Unknown spectator framing mode")
        self.mode=mode
        self.review=json.loads(review_path.read_text())
        require_strict_record_review(self.review)
        if Path(self.review["source_run"]).resolve()!=source.resolve():
            raise ValueError("Record review belongs to another episode")
        self.input_video=review_path.parent/"reviewer_view.mp4"
        self.input_hash=sha(self.input_video)
        if self.input_hash!=self.review["annotated_video_sha256"]:
            raise ValueError("Accepted source video hash mismatch")
        self.review_path=review_path;self.review_hash=sha(review_path)
        self.capture_source=cv2.VideoCapture(str(self.input_video))
        if abs(self.capture_source.get(cv2.CAP_PROP_FPS)-5)>1e-6:
            raise ValueError("Source playback rate must be 5 fps / 1x simulation time")
        self.frames=json.loads((source/"video_frames.json").read_text())
        self.events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
        self.result=dict(json.loads((source/"result.json").read_text()),
            reacquisition_step=self.review["temporal_evidence"]["reacquisition_step"])
        self.writer=imageio.get_writer(output/"reviewer_view.mp4",fps=5,codec="libx264",quality=8,
                                      macro_block_size=2,ffmpeg_log_level="error")
        self.records=[];self.panel_errors=[];self.closed=False

    def frame(self,io,step,phase):
        if step%4:return
        record=self.frames[len(self.records)]
        if record["sim_step"]!=step or record["phase"]!=phase:
            raise ValueError("Replay and original video are not step/phase aligned")
        ok,frame=self.capture_source.read()
        if not ok:raise ValueError("Accepted source video ended early")
        base=io.pose();tcp=array(io.robot.tcp_pose.p)[0]
        # A close overhead camera stays below the house ceiling and centers
        # the actual base and TCP, so a forward release need not leave frame.
        # Heading remains fixed in world coordinates as the robot rotates.
        eye,look,fovy=spectator_geometry(base,tcp,io.overview_heading,phase,self.mode)
        camera=io.base._human_render_cameras["render_camera"].camera
        camera.set_fovy(fovy)
        camera.set_local_pose(sapien_utils.look_at(eye,look).sp)
        rendered=array(io.base.render_rgb_array("render_camera"))[0].astype(np.uint8)
        if rendered.shape!=(540,960,3):raise ValueError("Unexpected spectator resolution")
        frame[36:576,:960]=cv2.cvtColor(rendered,cv2.COLOR_RGB2BGR)
        captions=logged_captions(step,self.result,self.events)
        for i,(caption,event_step) in enumerate(captions):
            top=46+34*i
            cv2.rectangle(frame,(12,top),(650,top+30),(204,235,245),-1)
            cv2.putText(frame,caption,(22,top+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(53,46,32),1,cv2.LINE_AA)
        self.writer.append_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.records.append(dict(sim_step=step,phase=phase,eye_world=eye.tolist(),look_world=look.tolist(),
            fovy_rad=fovy,tcp_world=tcp.tolist(),captions=[c for c,_ in captions]))

    def close(self):
        if self.closed:return
        self.closed=True;self.writer.close();self.capture_source.release()
        original=cv2.VideoCapture(str(self.input_video))
        rendered=cv2.VideoCapture(str(self.output/"reviewer_view.mp4"))
        decoded=0
        try:
            while True:
                ok,a=original.read();new_ok,b=rendered.read()
                if not ok or not new_ok:
                    complete=not ok and not new_ok;break
                self.panel_errors.append(float(np.abs(a[:,960:].astype(float)-b[:,960:].astype(float)).mean()))
                decoded+=1
        finally:original.release();rendered.release()
        checks=dict(all_original_frames_retained=complete and decoded==len(self.records)==len(self.frames),
            head_and_memory_panels_preserved=bool(self.panel_errors) and max(self.panel_errors)<12.,
            source_video_unchanged=sha(self.input_video)==self.input_hash,
            source_review_unchanged=sha(self.review_path)==self.review_hash)
        report=dict(checks=checks,render_checks_passed=all(checks.values()),frames=decoded,
            maximum_panel_mae=max(self.panel_errors,default=None),source_run=str(self.source.resolve()),
            original_review_sha256=self.review_hash,original_video_sha256=self.input_hash,
            reframed_video_sha256=sha(self.output/"reviewer_view.mp4"),renderer_script_sha256=sha(Path(__file__)),
            spectator_mode=getattr(self,"mode","overhead"),
            visual_acceptance_pending=True,release_ready=False,
            boundary="Alternate simulator-rendered spectator view of the SAME saved controls/forces at 1x. Recorded head/memory panels retained. Physical replay must independently match; this is not another policy trial.")
        (self.output/"spectator_camera_frames.json").write_text(json.dumps(self.records)+"\n")
        (self.output/"spectator_video_review.json").write_text(json.dumps(report,indent=2)+"\n")
        return report
