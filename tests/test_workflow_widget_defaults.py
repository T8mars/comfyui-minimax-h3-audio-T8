"""Old workflow bytes may omit appended optional widgets, never required ones."""
import pytest
from comfy_api.latest import io

from helpers import plugin_widget_map


class OptionalTail(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TestOptionalTail", inputs=[
            io.Float.Input("required", default=1.),
            io.Combo.Input("tail", options=["legacy", "explicit"], default="legacy", optional=True),
        ], outputs=[])


class NoOptionalDefault(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TestNoOptionalDefault", inputs=[
            io.Float.Input("required", default=1.), io.Float.Input("tail", optional=True),
        ], outputs=[])


def test_old_save_uses_only_declared_optional_default_without_rewriting_it():
    node = {"widgets_values": [2.]}
    assert plugin_widget_map(node, OptionalTail) == {"required": 2., "tail": "legacy"}
    assert node == {"widgets_values": [2.]}


def test_explicit_new_widget_still_overrides_default():
    assert plugin_widget_map({"widgets_values": [2., "explicit"]}, OptionalTail)["tail"] == "explicit"


def test_missing_required_is_still_an_error_even_with_a_declared_default():
    with pytest.raises(AssertionError, match="required"):
        plugin_widget_map({"widgets_values": []}, OptionalTail)


def test_missing_optional_without_default_is_not_invented():
    with pytest.raises(AssertionError, match="tail"):
        plugin_widget_map({"widgets_values": [2.]}, NoOptionalDefault)


def test_unexpected_trailing_values_still_fail():
    with pytest.raises(AssertionError, match="not described"):
        plugin_widget_map({"widgets_values": [2., "legacy", "unexpected"]}, OptionalTail)
