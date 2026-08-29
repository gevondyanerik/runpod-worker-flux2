"""The generated ComfyUI graph.

These tests are the reason the graph can be built in code rather than pasted
from the UI: the shape is pinned here and against golden files, so an
accidental rewiring fails on a laptop instead of on a paying request.
"""

from __future__ import annotations

import pytest

from app import workflow
from app.variants import VARIANTS, get_variant


def build(**overrides: object) -> dict:
    kwargs: dict = {
        "variant": get_variant("klein-4b"),
        "prompt": "a red bicycle",
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "steps": 4,
        "guidance": 1.0,
    }
    kwargs.update(overrides)
    return workflow.build(**kwargs)  # type: ignore[arg-type]


def test_text_to_image_graph_validates() -> None:
    graph = build()
    workflow.validate(graph, expected_references=0)
    assert graph[workflow.N_GUIDER]["inputs"]["positive"] == [workflow.N_PROMPT, 0]
    assert graph[workflow.N_GUIDER]["inputs"]["negative"] == [workflow.N_NEGATIVE, 0]
    assert workflow.N_REF_SIZE not in graph


def test_profile_drives_every_model_input() -> None:
    for name, variant in VARIANTS.items():
        graph = build(variant=variant, steps=variant.sampling.steps)
        assert graph["unet"]["inputs"]["unet_name"] == variant.diffusion.filename, name
        assert graph["clip"]["inputs"]["clip_name"] == variant.text_encoder.filename
        assert graph["vae"]["inputs"]["vae_name"] == variant.vae.filename
        assert graph["sampler"]["inputs"]["sampler_name"] == variant.sampling.sampler


@pytest.mark.parametrize("count", [1, 2, 6])
def test_reference_chain_is_connected(count: int) -> None:
    names = tuple(f"ref{i}.png" for i in range(count))
    graph = build(reference_filenames=names)
    workflow.validate(graph, expected_references=count)

    # Each reference conditions both branches and chains onto the previous one,
    # so nothing the caller sent is silently dropped.
    for index in range(count):
        assert graph[f"ref{index}_load"]["inputs"]["image"] == names[index]
        previous_pos = [f"ref{index - 1}_pos", 0] if index else [workflow.N_PROMPT, 0]
        assert graph[f"ref{index}_pos"]["inputs"]["conditioning"] == previous_pos
    assert graph[workflow.N_GUIDER]["inputs"]["positive"] == [f"ref{count - 1}_pos", 0]


def test_reference_order_is_preserved() -> None:
    # Order is semantic: a prompt may say "the product in image 1".
    names = ("first.png", "second.png", "third.png")
    graph = build(reference_filenames=names)
    for index, name in enumerate(names):
        assert graph[f"ref{index}_load"]["inputs"]["image"] == name


def test_reference_scale_uses_mebipixels_not_megapixels() -> None:
    # ImageScaleToTotalPixels multiplies by 1024*1024. Dividing the budget by
    # 1e6 would push every reference about 5% over the intended size.
    variant = get_variant("klein-4b")
    graph = build(variant=variant, reference_filenames=("a.png",))
    megapixels = graph["ref0_scale"]["inputs"]["megapixels"]
    assert megapixels == pytest.approx(variant.ref_max_pixels / (1024 * 1024))
    assert megapixels == 1.0


def test_edit_without_an_explicit_size_follows_the_reference() -> None:
    graph = build(reference_filenames=("a.png",), match_reference_size=True)
    workflow.validate(graph, expected_references=1)
    assert workflow.N_REF_SIZE in graph
    assert graph[workflow.N_REF_SIZE]["inputs"]["image"] == ["ref0_scale", 0]
    assert graph["latent"]["inputs"]["width"] == [workflow.N_REF_SIZE, 0]
    assert graph["latent"]["inputs"]["height"] == [workflow.N_REF_SIZE, 1]
    assert graph["sigmas"]["inputs"]["width"] == [workflow.N_REF_SIZE, 0]


def test_an_explicit_size_always_wins() -> None:
    graph = build(width=768, height=512, reference_filenames=("a.png",))
    assert workflow.N_REF_SIZE not in graph
    assert graph["latent"]["inputs"]["width"] == 768
    assert graph["latent"]["inputs"]["height"] == 512


def test_reference_size_is_ignored_without_references() -> None:
    graph = build(match_reference_size=True)
    assert workflow.N_REF_SIZE not in graph
    assert graph["latent"]["inputs"]["width"] == 1024


def test_dimensions_are_aligned_inside_the_builder() -> None:
    graph = build(width=1000, height=1000)
    assert graph["latent"]["inputs"]["width"] == 992
    assert graph["sigmas"]["inputs"]["width"] == 992


def test_batch_size_reaches_the_latent() -> None:
    assert build(batch_size=3)["latent"]["inputs"]["batch_size"] == 3


def test_validate_catches_a_dangling_link() -> None:
    graph = build()
    graph["decode"]["inputs"]["samples"] = ["nonexistent", 0]
    with pytest.raises(workflow.WorkflowInvalidError, match="missing node"):
        workflow.validate(graph, expected_references=0)


def test_validate_catches_a_missing_output() -> None:
    graph = build()
    del graph[workflow.N_SAVE]
    with pytest.raises(workflow.WorkflowInvalidError, match="output node"):
        workflow.validate(graph, expected_references=0)


def test_validate_catches_an_unwired_reference() -> None:
    graph = build(reference_filenames=("a.png", "b.png"))
    graph[workflow.N_GUIDER]["inputs"]["positive"] = ["ref0_pos", 0]
    with pytest.raises(workflow.WorkflowInvalidError, match="not connected"):
        workflow.validate(graph, expected_references=2)


def test_validate_catches_a_reference_count_mismatch() -> None:
    graph = build(reference_filenames=("a.png",))
    with pytest.raises(workflow.WorkflowInvalidError, match="expected 2"):
        workflow.validate(graph, expected_references=2)


def test_the_prompt_appears_exactly_once() -> None:
    graph = build(prompt="a very specific prompt")
    occurrences = [
        node_id
        for node_id, node in graph.items()
        if node["inputs"].get("text") == "a very specific prompt"
    ]
    assert occurrences == [workflow.N_PROMPT]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1_048_576, 1.0), (524_288, 0.5), (1_000, 0.01), (10**9, 16.0)],
)
def test_megapixels_stay_inside_the_node_range(value: int, expected: float) -> None:
    assert workflow._megapixels(value) == expected
