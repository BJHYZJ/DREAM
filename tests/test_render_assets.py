import hashlib
import json
import struct

from dream_sim.render_assets import build_cache, repair_uv_index


def glb(attributes):
    document = dict(
        meshes=[dict(primitives=[dict(attributes=attributes, material=0)])],
        materials=[dict(pbrMetallicRoughness=dict(baseColorTexture=dict(index=0, texCoord=1)))],
    )
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    binary = struct.pack("<II", 8, 0x004E4942) + b"geometry"
    return (
        b"glTF"
        + struct.pack("<IIII", 2, 20 + len(content) + len(binary), len(content), 0x4E4F534A)
        + content
        + binary
    )


def test_uv_alias_preserves_geometry_buffers_and_material_uv_selection():
    original = glb(dict(POSITION=0, NORMAL=1, TEXCOORD_1=2))
    output, changes = repair_uv_index(original)
    before_length = struct.unpack_from("<I", original, 12)[0]
    after_length = struct.unpack_from("<I", output, 12)[0]
    assert original[20 + before_length :] == output[20 + after_length :]
    before = json.loads(original[20 : 20 + before_length])
    after = json.loads(output[20 : 20 + after_length])
    assert before["materials"] == after["materials"]
    attributes = after["meshes"][0]["primitives"][0]["attributes"]
    assert attributes["POSITION"] == 0 and attributes["NORMAL"] == 1
    assert attributes["TEXCOORD_0"] == attributes["TEXCOORD_1"] == 2
    assert len(changes) == 1
    assert repair_uv_index(output) == (output, [])


def test_derived_cache_preserves_upstream_and_binds_both_hashes(tmp_path):
    source = tmp_path / "original"
    path = source / "data/scene_datasets/lamp.glb"
    path.parent.mkdir(parents=True)
    payload = glb(dict(POSITION=0, TEXCOORD_1=2))
    path.write_bytes(payload)
    upstream = hashlib.sha256(payload).hexdigest()
    lock = tmp_path / "upstream.json"
    lock.write_text(
        json.dumps(
            dict(
                complete=True,
                failures=[],
                artifacts={"lamp.glb": dict(bytes=len(payload), sha256=upstream)},
            )
        )
    )
    output = tmp_path / "render"
    output_lock = tmp_path / "render.json"
    result = build_cache(source, output, lock, output_lock)
    assert result["changed_assets"] == 1 and path.read_bytes() == payload
    derived = output / "data/scene_datasets/lamp.glb"
    manifest = json.loads(output_lock.read_text())
    assert (
        manifest["artifacts"]["lamp.glb"]["sha256"]
        == hashlib.sha256(derived.read_bytes()).hexdigest()
        != upstream
    )
    change = manifest["render_compatibility"]["changes"][0]
    assert change["source_sha256"] == upstream and change["binary_buffers_unchanged"]
