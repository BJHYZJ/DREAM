"""Synthetic portability checks only; these are not simulated task outcomes."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import run_instruction_profile as profiles
from build_instruction_repro_profiles import copy_checked, source_id


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data)+"\n")


def catalog(tmp_path):
    root=tmp_path/"SYNTHETIC_profiles";source=root/"source/DREAM_code"
    entrypoint=source/"experiments/run_instruction_task.py"
    entrypoint.parent.mkdir(parents=True);entrypoint.write_text("# SYNTHETIC, NEVER EXECUTE\n")
    hashes={"experiments/run_instruction_task.py":profiles.sha(entrypoint)}
    put(root/"source/manifest.json",hashes)
    entry=dict(repository="source/DREAM_code",manifest="source/manifest.json",
        manifest_sha256=profiles.sha(root/"source/manifest.json"))
    locks={}
    for key in ("model_lock","asset_lock"):
        put(root/f"{key}.json",dict(synthetic=True))
        locks[key]=dict(file=f"{key}.json",sha256=profiles.sha(root/f"{key}.json"))
    cases=[]
    for n in range(10):
        key=f"{n+1:02d}";folder=root/"tasks"/key
        task=dict(scene=f"SYNTHETIC-house-{n}",seed=100+n,recipe=dict(id=f"recipe-{n%5}"),
            instruction=f"SYNTHETIC instruction {n}",room_map_file="room.npz")
        put(folder/"task.json",task);(folder/"room.npz").write_bytes(b"SYNTHETIC map")
        config=dict(threshold=.15,navigation_budget=110,variant="dynamic",video=True)
        if n>=8:config["heading_navigation"]=True
        cases.append(dict(id=key,scene=task["scene"],seed=task["seed"],recipe=task["recipe"]["id"],
            instruction=task["instruction"],source_id="synthetic-source",task=f"tasks/{key}/task.json",
            task_sha256=profiles.sha(folder/"task.json"),room_map_sha256=profiles.sha(folder/"room.npz"),
            configuration=config))
    put(root/"recorded_protocol.json",dict(synthetic=True))
    data=dict(schema_version=1,scope="versioned_instruction_reproduction_profiles",cases=cases,
        sources={"synthetic-source":entry},**locks,
        study=dict(source_id="synthetic-source",case_order=[c["id"] for c in cases],
            seeds=[100,101,102],variants=["dynamic","static"],heading_navigation=True,
            gpu_workers=["0","1","0","1","0","1"],recorded_protocol="recorded_protocol.json",
            recorded_protocol_sha256=profiles.sha(root/"recorded_protocol.json")))
    put(root/"profiles.json",data)
    return root,root/"profiles.json",data


@pytest.mark.parametrize("case_id,heading",[("01",False),("09",True),("10",True)])
def test_historical_profile_is_a_dry_run_with_its_own_heading_setting(tmp_path,case_id,heading):
    root,path,_=catalog(tmp_path)
    command,report=profiles.build_command(path,case_id=case_id,output=tmp_path/"new_attempt",
        python="SYNTHETIC_python",launcher="SYNTHETIC_launcher")
    assert ("--heading-navigation" in command)==heading
    assert report["case_ids"]==[case_id] and report["policy_executed"] is False
    assert report["asset_contents_verified_by_this_launcher"] is False
    assert "--benchmark" not in command and not (tmp_path/"new_attempt").exists()
    assert str(root/"tasks"/case_id/"task.json") in command


def test_study_retains_all_cases_seeds_variants_and_one_source(tmp_path):
    root,path,data=catalog(tmp_path)
    command,report=profiles.build_command(path,study=True,output=tmp_path/"new_study",gpus=["0"])
    assert len(report["case_ids"])==10 and "--benchmark" in command and "--heading-navigation" in command
    assert command[command.index("--seeds")+1:command.index("--variants")]==["100","101","102"]
    assert command[command.index("--variants")+1:]==["dynamic","static"]
    assert command[command.index("--gpus")+1]=="0"
    assert report["source_id"]==data["study"]["source_id"]


@pytest.mark.parametrize("change",["duplicate_house","fewer_recipes","source_manifest","model_lock","asset_lock","protocol"])
def test_changed_catalog_inputs_are_rejected(tmp_path,change):
    root,path,data=catalog(tmp_path)
    if change=="duplicate_house":data["cases"][1]["scene"]=data["cases"][0]["scene"]
    elif change=="fewer_recipes":
        for case in data["cases"]:case["recipe"]="one-recipe"
    else:
        target={"source_manifest":"source/manifest.json","protocol":"recorded_protocol.json"}.get(change,change+".json")
        (root/target).write_text("CHANGED")
    put(path,data)
    with pytest.raises(ValueError):profiles.load_catalog(path)


@pytest.mark.parametrize("change",["source_bytes","extra_module","source_symlink","task_bytes","room_bytes","threshold","navigation_budget","variant","video"])
def test_source_task_and_unsupported_parameters_fail_before_execution(tmp_path,change):
    root,path,data=catalog(tmp_path)
    if change=="source_bytes":(root/"source/DREAM_code/experiments/run_instruction_task.py").write_text("changed")
    elif change=="extra_module":(root/"source/DREAM_code/experiments/extra.py").write_text("# additional")
    elif change=="source_symlink":
        original=root/"source/DREAM_code/experiments/run_instruction_task.py"
        target=root/"source/DREAM_code/original.txt";original.rename(target);original.symlink_to(target)
    elif change in ("task_bytes","room_bytes"):
        (root/"tasks/01"/("task.json" if change=="task_bytes" else "room.npz")).write_text("{}")
    else:data["cases"][0]["configuration"][change]="unsupported"
    put(path,data)
    with pytest.raises(ValueError):profiles.build_command(path,case_id="01",output=tmp_path/"new")


@pytest.mark.parametrize("change",["unknown_case","duplicate_order","seeds","variants","heading"])
def test_study_design_cannot_silently_drift(tmp_path,change):
    _,path,data=catalog(tmp_path)
    if change=="unknown_case":data["study"]["case_order"][0]="99"
    if change=="duplicate_order":data["study"]["case_order"][0]="02"
    if change=="seeds":data["study"]["seeds"]=[1,2,3]
    if change=="variants":data["study"]["variants"]=["dynamic"]
    if change=="heading":data["study"]["heading_navigation"]=False
    put(path,data)
    with pytest.raises(ValueError):profiles.build_command(path,study=True,output=tmp_path/"new")


def test_new_output_and_exactly_one_mode_required(tmp_path):
    _,path,_=catalog(tmp_path)
    with pytest.raises(FileExistsError):profiles.build_command(path,case_id="01",output=tmp_path)
    with pytest.raises(ValueError):profiles.build_command(path,output=tmp_path/"new")
    with pytest.raises(ValueError):profiles.build_command(path,case_id="01",study=True,output=tmp_path/"new")
    with pytest.raises(ValueError):profiles.build_command(path,case_id="99",output=tmp_path/"new")


def test_catalog_relative_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):profiles.child(tmp_path,"../outside")
    with pytest.raises(ValueError):profiles.child(tmp_path,str(tmp_path/"inside"))


def test_export_preserves_bytes_rejects_changed_digest_and_existing_target(tmp_path):
    source=tmp_path/"source.json";source.write_text('{"synthetic":true}\n')
    destination=tmp_path/"export/copy.json"
    assert copy_checked(source,destination,profiles.sha(source))==profiles.sha(destination)
    with pytest.raises(FileExistsError):copy_checked(source,destination)
    with pytest.raises(ValueError,match="Source changed"):copy_checked(source,tmp_path/"other.json","0"*64)
    assert not (tmp_path/"other.json").exists()


def test_export_secret_like_content_is_rejected_without_printing_it(tmp_path):
    source=tmp_path/"SYNTHETIC.json";token="hf_"+"a"*25;source.write_text(json.dumps(dict(fake=token)))
    with pytest.raises(ValueError) as error:copy_checked(source,tmp_path/"copy.json")
    assert token not in str(error.value) and not (tmp_path/"copy.json").exists()


def test_source_deduplication_uses_bytes_not_counts_or_dict_order():
    assert source_id({"a":"1","b":"2"})==source_id({"b":"2","a":"1"})
    assert source_id({"a":"1","b":"2"})!=source_id({"a":"3","b":"2"})
