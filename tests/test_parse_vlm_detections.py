import numpy as np
from PIL import Image

from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced


def _api() -> FrankaLiberoApiReduced:
    return FrankaLiberoApiReduced.__new__(FrankaLiberoApiReduced)


def _api_for_query_vlm() -> FrankaLiberoApiReduced:
    api = _api()
    api._vlm_backend = {}
    return api


def _patch_query_model(monkeypatch):
    calls = []

    def fake_query_model(_args, messages):
        calls.append(messages)
        return {"content": "ok"}

    monkeypatch.setattr("capx.llm.client.query_model", fake_query_model)
    return calls


def test_gpt_vapi_auto_prefers_pixel_coordinates(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_VLM_MODEL", "vapi/gpt-5.5")
    monkeypatch.delenv("CAPX_VLM_COORD_SPACE", raising=False)
    image = np.zeros((512, 800, 3), dtype=np.uint8)

    det = _api().parse_vlm_detections(
        '{"box":[500,211,622,324],"point":[561,268]}',
        image=image,
    )

    assert det["boxes"][0] == [500.0, 211.0, 622.0, 324.0]
    assert det["points"][0] == [561.0, 267.5]
    assert det["points"][1] == [561.0, 268.0]


def test_query_vlm_accepts_canonical_images_keyword(monkeypatch) -> None:
    calls = _patch_query_model(monkeypatch)
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    assert _api_for_query_vlm().query_vlm("is this visible?", images=image) == "ok"

    content = calls[0][0]["content"]
    assert content[0] == {"type": "text", "text": "is this visible?"}
    assert content[1]["type"] == "image_url"


def test_query_vlm_accepts_legacy_image_then_prompt(monkeypatch) -> None:
    calls = _patch_query_model(monkeypatch)
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    assert _api_for_query_vlm().query_vlm(image, "is this visible?") == "ok"

    content = calls[0][0]["content"]
    assert content[0] == {"type": "text", "text": "is this visible?"}
    assert content[1]["type"] == "image_url"


def test_query_vlm_accepts_image_keyword_and_path(monkeypatch, tmp_path) -> None:
    calls = _patch_query_model(monkeypatch)
    path = tmp_path / "crop.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(path)

    assert _api_for_query_vlm().query_vlm(prompt="classify this", image=str(path)) == "ok"
    assert _api_for_query_vlm().query_vlm(str(path), "classify this") == "ok"

    assert calls[0][0]["content"][0]["text"] == "classify this"
    assert calls[1][0]["content"][0]["text"] == "classify this"


def test_auto_defaults_to_pixel_coordinates(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_VLM_MODEL", raising=False)
    image = np.zeros((512, 800, 3), dtype=np.uint8)

    det = _api().parse_vlm_detections(
        '{"box":[500,211,622,324],"point":[561,268]}',
        image=image,
    )

    assert det["boxes"][0] == [500.0, 211.0, 622.0, 324.0]
    assert det["points"][0] == [561.0, 267.5]
    assert det["points"][1] == [561.0, 268.0]


def test_explicit_coord_space_overrides_backend(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_VLM_MODEL", "vapi/gpt-5.5")
    image = np.zeros((512, 800, 3), dtype=np.uint8)

    det = _api().parse_vlm_detections(
        '{"box":[500,211,622,324]}',
        image=image,
        coord_space="norm1000",
    )

    assert det["boxes"][0] == [400.0, 108.032, 497.6, 165.888]


def test_fraction_coordinates_still_scale_in_auto(monkeypatch) -> None:
    image = np.zeros((512, 800, 3), dtype=np.uint8)

    det = _api().parse_vlm_detections(
        '{"box":[0.5,0.25,0.75,0.5]}',
        image=image,
    )

    assert det["boxes"][0] == [400.0, 128.0, 600.0, 256.0]


def test_vlm_bbox_detection_rescales_qwen_norm1000(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_VLM_MODEL", "openrouter/qwen/qwen3.6-plus")
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    api = _api()
    api.query_vlm = lambda *args, **kwargs: '{"box":[500,250,1000,500]}'
    api._save_debug_overlay = lambda *args, **kwargs: None

    assert api.vlm_bbox_detection(image, "target") == [400.0, 100.0, 799.0, 200.0]


def test_vlm_bbox_detection_uses_instance_backend_config(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_VLM_MODEL", raising=False)
    monkeypatch.delenv("CAPX_VLM_COORD_SPACE", raising=False)
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    api = _api()
    api.configure_vlm_backend(model="openrouter/qwen/qwen3.6-plus")
    api.query_vlm = lambda *args, **kwargs: '{"box":[500,250,1000,500]}'
    api._save_debug_overlay = lambda *args, **kwargs: None

    assert api.vlm_bbox_detection(image, "target") == [400.0, 100.0, 799.0, 200.0]


def test_vlm_point_detection_rescales_qwen_norm1000(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_VLM_MODEL", "openrouter/qwen/qwen3.6-plus")
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    api = _api()
    api.query_vlm = lambda *args, **kwargs: '{"point":[500,250]}'
    api._save_debug_overlay = lambda *args, **kwargs: None

    assert api.vlm_point_detection(image, "target") == [400.0, 100.0]
