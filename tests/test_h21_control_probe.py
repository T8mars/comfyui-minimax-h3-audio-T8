"""Actual CPU Director preparation; graph-only fixture weights never execute."""

from copy import deepcopy
import asyncio
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import uuid

from PIL import Image
import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project


ROOT = Path(__file__).resolve().parents[1]
CORE = Path(importlib.util.find_spec('folder_paths').origin).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "h21_control_probe_tested", ROOT / "tools/h21_control_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def tiny_header(path):
    """A real safetensors file, deliberately not trained execution weights."""
    header = json.dumps(
        {
            "graph_only": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "__metadata__": {"purpose": "CPU graph preparation only, not inference"},
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, record_testsuite_property):
    base = tmp_path_factory.mktemp("owned-h21-cpu-probe")
    store = ProjectStore(base / "source-user", base / "source-input")
    asset_id = str(uuid.uuid4())
    image = store.input_root / "t8_director" / asset_id / "source.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (57, 93), (20, 80, 130)).save(image)
    media = store.register_asset(image, asset_id, "owned-control.png")
    project = new_project()
    project["assets"] = [media]
    shot = project["doc"]["shots"][0]
    shot.update(
        simplePrompt="A steady empty room, quiet room tone.",
        mode="text",
        sound="native",
        manualDuration=1.0,
        duration=1.0,
        autoDuration=False,
    )
    shot["controlRegions"] = {
        "version": 1,
        "items": [
            {
                "id": str(uuid.uuid4()),
                "asset_id": asset_id,
                "control_net_name": "custom-unlisted.safetensors",
                "strength": 0.7,
                "region": {"mode": "rect", "xyxy_normalized": [0.1, 0.2, 0.8, 0.9]},
                "crop": {"enabled": True, "xyxy_normalized": [-0.1, 0, 1.2, 1.0]},
            }
        ],
    }
    generation = project["doc"]["generation"]
    generation.update(
        unet="custom-unet.safetensors",
        clip="custom-clip.safetensors",
        video_vae="custom-video.safetensors",
        audio_vae="custom-audio.safetensors",
        lora_mode="none",
    )
    roots = {}
    for category in probe.MODEL_FOLDERS:
        directory = base / "readonly-models" / category
        directory.mkdir(parents=True)
        roots[category] = [str(directory)]
    for category, name in [
        ("diffusion_models", generation["unet"]),
        ("text_encoders", generation["clip"]),
        ("vae", generation["video_vae"]),
        ("vae", generation["audio_vae"]),
        ("model_patches", "custom-unlisted.safetensors"),
        ("latent_upscale_models", "custom-3d.safetensors"),
        ("hyperflow", "custom-hf.safetensors"),
    ]:
        tiny_header(Path(roots[category][0]) / name)
    project_path = base / "owned-project.json"
    project_path.write_text(json.dumps(project), encoding="utf-8")
    config = {
        "reserve_vram_gb": 5,
        "project": str(project_path),
        "assets": {asset_id: str(image)},
        "model_directories": roots,
        "seed": 26092101,
        "cases": [
            {
                "name": "single",
                "sampling": {
                    "mode": "single",
                    "resolution_mp": 0.2,
                    "lora_mode": "none",
                    "loras": [],
                },
            },
            {
                "name": "standard4plus4",
                "sampling": {
                    "mode": "two_pass",
                    "output_mp": 0.2,
                    "upscaler": "custom-3d.safetensors",
                    "low_loras": [],
                    "high_loras": [],
                },
            },
            {
                "name": "hf_continuous4plus4",
                "sampling": {
                    "mode": "hyperflow",
                    "variant": "continuous4plus4",
                    "hyperflow_file": "hyperflow/custom-hf.safetensors",
                    "output_mp": 0.2,
                    "low_loras": [],
                    "high_loras": [],
                },
            },
            {
                "name": "baseline_single",
                "sampling": {
                    "mode": "single",
                    "resolution_mp": 0.2,
                    "lora_mode": "none",
                    "loras": [],
                },
                "controlRegions": {"version": 1, "items": []},
            },
        ],
    }
    config_path = base / "owned-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = {
        p: probe.file_identity(p)
        for directory in roots.values()
        for p in Path(directory[0]).iterdir()
    }
    run = base / "private-run"
    manifest = probe.prepare(config_path, run, CORE, python=sys.executable)
    record_testsuite_property("prepared_manifest", str(run / "manifest.json"))
    record_testsuite_property("prepared_config", str(config_path))
    return base, run, config_path, config, project, manifest, before


def test_real_cpu_prepare_three_recipes_no_server_gpu_or_weight_execution(prepared):
    _, run, _, _, _, manifest, weights = prepared
    assert manifest["stage"] == "prepared_cpu_graphs_only"
    assert manifest["gpu_started"] is False and manifest["model_execution"] is False
    assert manifest["cuda_initialized"] is False
    assert manifest["core_validate_prompt"] is True
    assert all(
        probe.read_json(run / case["validation"])["valid"] is True
        for case in manifest["cases"]
    )
    assert [case["name"] for case in manifest["cases"]] == [
        "single",
        "standard4plus4",
        "hf_continuous4plus4",
        "baseline_single",
    ]
    assert (
        not (run / "execute-started.json").exists()
        and not (run / "server.stdout.log").exists()
    )
    assert all(
        probe.file_identity(path) == identity for path, identity in weights.items()
    )
    assert probe.verify_prepared(run / "manifest.json")["run_id"] == manifest["run_id"]


def test_raw_control_rgb_bytes_coordinates_models_and_native_recipe_preserved(prepared):
    _, run, _, _, source, manifest, _ = prepared
    asset = source["assets"][0]
    copied = run / "input/t8_director" / asset["id"] / "source.png"
    assert probe.file_identity(copied)["sha256"] == asset["sha256"]
    with Image.open(copied) as image:
        assert image.size == (57, 93)
    assert not list(copied.parent.glob("fit_*"))
    for case in manifest["cases"]:
        built = probe.read_json(run / case["built"])
        graph = built["prompt"]
        assert graph["1"]["inputs"]["unet_name"] == "custom-unet.safetensors"
        assert graph["4"]["inputs"]["clip_name"] == "custom-clip.safetensors"
        if case["name"] == "baseline_single":
            assert not any(
                node["class_type"] == "MiniMaxH3DirectorControlSourceEXPT8"
                for node in graph.values()
            )
            assert probe.read_json(run / case["project"])["doc"]["shots"][0][
                "controlRegions"
            ] == {"version": 1, "items": []}
            continue
        assert any(
            node["class_type"] == "LoadImage"
            and node["inputs"]["image"] == asset["server_path"]
            for node in graph.values()
        )
        sources = [
            node
            for node in graph.values()
            if node["class_type"] == "MiniMaxH3DirectorControlSourceEXPT8"
        ]
        assert sources
        for node in sources:
            spec = json.loads(node["inputs"]["spec_json"])
            assert spec["crop"]["xyxy_normalized"] == [-0.1, 0, 1.2, 1]
            assert spec["region"]["xyxy_normalized"] == [0.1, 0.2, 0.8, 0.9]
        assert all(row["asset_id"] != asset["id"] for row in built["shot"]["media_map"])
    standard = probe.read_json(run / manifest["cases"][1]["prompt"])
    assert any(
        node["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced"
        for node in standard.values()
    )
    assert (
        sum(node["class_type"] == "SamplerCustomAdvanced" for node in standard.values())
        == 2
    )


def test_source_freeze_is_small_runtime_allowlist_not_research_weights_or_user_data(
    prepared,
):
    _, run, _, _, _, manifest, _ = prepared
    package = Path(manifest["package_root"])
    assert package.name.startswith("t8_h21_") and package.name != ROOT.name
    paths = [
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    ]
    assert "__init__.py" in paths and "h3_t8/director_generation.py" in paths
    assert set(paths) == set(manifest["runtime_allowlist"])
    assert all(
        Path(p).suffix == ".py"
        or p.startswith("web/")
        and Path(p).suffix in probe.STATIC_SUFFIXES
        for p in paths
    )
    assert "sol_attn_minimax_v2.py" in paths
    assert not any(
        p.startswith(("artifacts/", "tests/", "models/"))
        or p in {"SKILL.md", "ROADMAP.md", "meta.json", "features.json"}
        for p in paths
    )
    assert len(manifest["builder_imports"]) > 10
    for identity in manifest["builder_imports"].values():
        assert Path(identity["path"]).is_relative_to(package)
        assert not Path(identity["path"]).is_relative_to(ROOT)
    config = probe.read_json(run / "paths.json")
    assert config["h21_nodes"]["custom_nodes"] == str(run / "runtime_nodes")
    assert manifest["resources"]["models"][
        "fun_control_selection:custom-unlisted.safetensors"
    ]["path"].endswith("model_patches\\custom-unlisted.safetensors") or manifest[
        "resources"
    ]["models"]["fun_control_selection:custom-unlisted.safetensors"]["path"].endswith(
        "model_patches/custom-unlisted.safetensors"
    )


def test_server_argv_private_paths_unique_whitelist_memory_db_and_no_core_input(
    prepared,
):
    _, run, _, _, _, plan, _ = prepared
    command = probe.server_command(plan)
    for flag, folder in [
        ("--input-directory", "input"),
        ("--output-directory", "output"),
        ("--temp-directory", "temp"),
        ("--user-directory", "user"),
        ("--models-directory", "models"),
    ]:
        assert command[command.index(flag) + 1] == str(run / folder)
        assert str(CORE / folder) not in command
    first, last = (
        command.index("--whitelist-custom-nodes") + 1,
        command.index("--extra-model-paths-config"),
    )
    assert command[first:last] == [plan["package_name"], plan["identity_extension"]]
    assert command[command.index("--database-url") + 1] == "sqlite:///:memory:"
    assert "--disable-comfy-compiler" in command and "--disable-auto-launch" in command
    assert command[command.index("--reserve-vram") + 1] == "5"
    assert command[command.index("--listen") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] != "8189"


def test_prepare_evidence_never_overwritten(prepared):
    _, run, config_path, _, _, _, _ = prepared
    before = (run / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="never overwritten"):
        probe.prepare(config_path, run, CORE)
    assert (run / "manifest.json").read_bytes() == before


def test_actual_core_categories_preserved_without_undeclared_model_roots(prepared):
    _, run, _, config, _, plan, _ = prepared
    effective = plan["effective_model_directories"]
    assert effective["vae_approx"]["paths"] == []
    assert effective["vae_approx"]["extensions"]
    assert effective["checkpoints"]["paths"] == []
    declared = {
        Path(path).resolve()
        for roots in config["model_directories"].values()
        for path in roots
    }
    for value in effective.values():
        for path in value["paths"]:
            actual = Path(path).resolve()
            assert actual in declared or actual.is_relative_to(run)
            assert not actual.is_relative_to(CORE / "models")
            assert not actual.is_relative_to(CORE / "input")
            assert not actual.is_relative_to(CORE / "user")


def test_model_symlink_declared_and_real_identity_preserved(tmp_path):
    original = tmp_path / "actual.safetensors"
    tiny_header(original)
    linked = tmp_path / "declared.safetensors"
    try:
        linked.symlink_to(original)
    except OSError as error:
        pytest.skip(f"Host does not allow an owned model symlink: {error}")
    identity = probe.file_identity(linked)
    assert identity["declared_path"] == str(linked.absolute())
    assert identity["path"] == str(original.resolve())
    assert identity["sha256"] == probe.file_identity(original)["sha256"]


def test_runtime_source_symlink_rejected_without_copying_private_tree(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "h3_t8").mkdir()
    private = tmp_path / "never-copy-private"
    private.mkdir()
    (private / "secret.py").write_text("private = True", encoding="utf-8")
    linked = root / "web"
    try:
        linked.symlink_to(private, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Host does not allow an owned source symlink: {error}")
    with pytest.raises(ValueError, match="must not be symlinks"):
        probe.runtime_sources(root)


def test_changed_frozen_source_and_added_runtime_file_rejected_before_execute(prepared):
    _, run, _, _, _, plan, _ = prepared
    path = Path(plan["package_root"]) / "h3_t8/director_generation.py"
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\n# test tamper\n")
        with pytest.raises(RuntimeError, match="Prepared file changed"):
            probe.verify_prepared(run / "manifest.json")
    finally:
        path.write_bytes(before)
    extra = Path(plan["package_root"]) / "unexpected.py"
    try:
        extra.write_text('raise RuntimeError("not frozen")', encoding="utf-8")
        with pytest.raises(RuntimeError, match="gained or lost"):
            probe.verify_prepared(run / "manifest.json")
    finally:
        extra.unlink()


def test_selected_weight_change_is_real_error_without_model_name_whitelist(prepared):
    _, run, _, _, _, plan, _ = prepared
    model = next(iter(plan["resources"]["models"].values()))
    path = Path(model["path"])
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"x")
        with pytest.raises(RuntimeError, match="model bytes changed"):
            probe.verify_prepared(run / "manifest.json")
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "C:/other", "a/../../other"])
def test_relative_output_escape_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        probe.inside(tmp_path, bad)


def test_explicit_model_alias_roots_and_unused_empty_categories(prepared):
    _, _, _, config, _, _, _ = prepared
    roots = deepcopy(config["model_directories"])
    roots["clip"] = roots.pop("text_encoders")
    assert (
        probe.model_directories(roots)["text_encoders"]
        == config["model_directories"]["text_encoders"]
    )
    roots["text_encoders"] = []
    with pytest.raises(ValueError, match="same ordered roots"):
        probe.model_directories(roots)
    with pytest.raises(ValueError, match="Explicit"):
        probe.model_directories({"diffusion_models": []})


def test_8189_and_user_input_roots_rejected_without_creating_any_target(prepared):
    base, _, config_path, _, _, _, _ = prepared
    root = base / "never-created"
    with pytest.raises(ValueError, match="8189"):
        probe.prepare(config_path, root, CORE, port=8189)
    assert not root.exists()
    with pytest.raises(ValueError, match="user Core data"):
        probe.prepare(config_path, CORE / "input" / ("never-" + uuid.uuid4().hex), CORE)


def test_no_execute_default_and_internal_cli_cannot_accidentally_run_without_config():
    with pytest.raises(SystemExit) as error:
        probe.main([])
    assert error.value.code == 2


def test_case_selection_requires_explicit_deliberate_subset(prepared):
    _, _, _, _, _, plan, _ = prepared
    with pytest.raises(ValueError, match="select-case"):
        probe.select_cases(plan)
    assert [case["name"] for case in probe.select_cases(plan, ["single"])] == ["single"]
    assert probe.select_cases(plan, all_cases=True) == plan["cases"]
    for names in [["missing"], ["single", "single"]]:
        with pytest.raises(ValueError):
            probe.select_cases(plan, names)
    with pytest.raises(SystemExit) as error:
        probe.main(["--execute", "missing.json", "--dry-run"])
    assert error.value.code == 2


@pytest.fixture
def core_loaded_identity(prepared, monkeypatch):
    """Core's real path-named package loader; no listener/service or GPU queue."""
    import folder_paths
    import nodes
    from aiohttp import web
    from comfy_extras import nodes_custom_sampler
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import GetVideoComponents, LoadVideo
    from server import PromptServer

    _, run, _, _, _, plan, _ = prepared
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        deepcopy(folder_paths.folder_names_and_paths),
    )
    monkeypatch.setattr(folder_paths, "models_dir", str(run / "models"))
    for key in (
        "input_directory",
        "output_directory",
        "temp_directory",
        "user_directory",
    ):
        monkeypatch.setattr(folder_paths, key, getattr(folder_paths, key))
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", dict(nodes.NODE_CLASS_MAPPINGS))
    monkeypatch.setattr(
        nodes, "NODE_DISPLAY_NAME_MAPPINGS", dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    )
    monkeypatch.setattr(nodes, "EXTENSION_WEB_DIRS", dict(nodes.EXTENSION_WEB_DIRS))
    monkeypatch.setattr(nodes, "LOADED_MODULE_DIRS", dict(nodes.LOADED_MODULE_DIRS))
    # Actual aiohttp route table; only the PromptServer route container exists.
    # Constructing a real PromptServer would add unrelated database/frontend state.
    monkeypatch.setattr(
        PromptServer,
        "instance",
        SimpleNamespace(routes=web.RouteTableDef()),
        raising=False,
    )
    probe._configure(plan, cpu=True)
    for cls in (
        nodes_custom_sampler.BasicGuider,
        nodes_custom_sampler.RandomNoise,
        nodes_custom_sampler.SamplerCustomAdvanced,
        LoadVideo,
        GetVideoComponents,
        PreviewAny,
    ):
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    package = Path(plan["package_root"])
    assert asyncio.run(nodes.load_custom_node(str(package))) is True
    actual_name = str(package).replace(".", "_x_")
    assert actual_name != plan["package_name"]
    assert plan["package_name"] not in sys.modules
    owner = sys.modules[actual_name]
    assert Path(owner.__file__).resolve() == package / "__init__.py"
    extension_root = run / "runtime_nodes" / plan["identity_extension"]
    assert asyncio.run(nodes.load_custom_node(str(extension_root))) is True
    extension = sys.modules[str(extension_root).replace(".", "_x_")]
    return plan, nodes, owner, extension


def test_actual_core_loader_identity_accepts_path_namespace_and_same_object_alias(
    core_loaded_identity, monkeypatch
):
    plan, _, owner, extension = core_loaded_identity
    monkeypatch.setitem(sys.modules, "h21_same_object_alias_" + uuid.uuid4().hex, owner)
    response = asyncio.run(extension.identity(None))
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["ready"] is True, payload
    assert payload["run_id"] == plan["run_id"]
    assert payload["errors"] == []
    assert payload["loaded_namespace"] == owner.__name__


def test_actual_core_loader_identity_missing_owner_returns_negative_json(
    core_loaded_identity, monkeypatch
):
    _, _, owner, extension = core_loaded_identity
    for name, module in list(sys.modules.items()):
        if module is owner:
            monkeypatch.delitem(sys.modules, name)
    response = asyncio.run(extension.identity(None))
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["ready"] is False
    assert "missing_frozen_runtime_owner" in payload["errors"]


def test_actual_core_loader_identity_foreign_class_returns_negative_json(
    core_loaded_identity, monkeypatch
):
    _, nodes, _, extension = core_loaded_identity
    monkeypatch.setitem(
        nodes.NODE_CLASS_MAPPINGS,
        "MiniMaxH3DirectorControlSourceEXPT8",
        nodes.LoadImage,
    )
    response = asyncio.run(extension.identity(None))
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["ready"] is False
    assert "foreign_class:MiniMaxH3DirectorControlSourceEXPT8" in payload["errors"]
