"""Load-path Gradio packer: merge per-tab named slot dicts."""

from __future__ import annotations

import html
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gradio as gr

from ..assistant import build_analysis_brief_from_session
from ..loaders import FORMAT_LABELS
from ..session import LoadError, LoadedSession, load_session
from . import overview_tab, patterns_tab, raw_tab, upload, workflow_tab
from .shared import SharedState

PackFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class LoadContext:
    """Layout refs handed to each registry unit when wiring Gradio outputs."""

    main_tabs: Any
    shared: SharedState
    refs: Mapping[object, Any]


@dataclass(frozen=True)
class LoadUnit:
    """One load contributor: packed values and the Gradio components they fill."""

    name: str
    pack: PackFn
    slots: Callable[[LoadContext], dict[str, Any]]
    module: Any = None


def pack_shell(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict[str, Any]:
    """Tabs visibility and Gradio state filled on load."""
    del dark, banner
    if session is None:
        return {
            "main_tabs": gr.update(visible=False),
            "state_steps": [],
            "state_raw": {},
            "state_analysis_brief": "",
        }
    return {
        "main_tabs": gr.update(visible=True),
        "state_steps": session.steps,
        "state_raw": session.raw,
        "state_analysis_brief": build_analysis_brief_from_session(session),
    }


def _shell_slots(ctx: LoadContext) -> dict[str, Any]:
    return {
        "main_tabs": ctx.main_tabs,
        "state_steps": ctx.shared.state_steps,
        "state_raw": ctx.shared.state_raw,
        "state_analysis_brief": ctx.shared.state_analysis_brief,
    }


def _tab_unit(mod: Any) -> LoadUnit:
    """Bind a tab module's `pack_load` and `load_slots` as one registry entry."""

    def slots(ctx: LoadContext, tab: Any = mod) -> dict[str, Any]:
        return tab.load_slots(ctx.refs[tab])

    name = str(getattr(mod, "__name__", type(mod).__name__)).rsplit(".", 1)[-1]
    return LoadUnit(name=name, pack=mod.pack_load, slots=slots, module=mod)


LOAD_UNITS: tuple[LoadUnit, ...] = (
    LoadUnit("shell", pack_shell, _shell_slots),
    _tab_unit(upload),
    _tab_unit(overview_tab),
    _tab_unit(patterns_tab),
    _tab_unit(workflow_tab),
    _tab_unit(raw_tab),
)


def _ref_label(obj: object) -> str:
    name = getattr(obj, "__name__", None)
    if isinstance(name, str):
        return name.rsplit(".", 1)[-1]
    return repr(obj)


def _registered_tab_modules() -> frozenset[Any]:
    return frozenset(unit.module for unit in LOAD_UNITS if unit.module is not None)


def merge_packer_dicts(parts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Union packer dicts; duplicate keys are a bug."""
    packed: dict[str, Any] = {}
    for part in parts:
        overlap = packed.keys() & part.keys()
        if overlap:
            raise ValueError(f"Duplicate load slot keys: {sorted(overlap)}")
        packed.update(part)
    return packed


def collect_pack(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict[str, Any]:
    """Run every registered packer and merge."""
    return merge_packer_dicts([unit.pack(session, dark=dark, banner=banner) for unit in LOAD_UNITS])


_LOAD_SLOT_KEYS: frozenset[str] | None = None


def load_slot_keys() -> frozenset[str]:
    """Union of keys every packer emits (empty load)."""
    global _LOAD_SLOT_KEYS
    if _LOAD_SLOT_KEYS is None:
        _LOAD_SLOT_KEYS = frozenset(collect_pack())
    return _LOAD_SLOT_KEYS


def merge_load_slots(*, main_tabs, shared: SharedState, refs: Mapping[object, Any]) -> dict[str, Any]:
    """Component map for `do_load`.

    A new load-backed tab adds `pack_load` + `load_slots` on the tab module,
    one `_tab_unit(tab)` in `LOAD_UNITS`, and `refs[tab] = tab.layout()` in
    the Blocks composer.
    """
    expected = _registered_tab_modules()
    got = frozenset(refs)
    if got != expected:
        raise ValueError(
            "Load tab refs mismatch vs registry: "
            f"missing={sorted(_ref_label(m) for m in expected - got)} "
            f"extra={sorted(_ref_label(m) for m in got - expected)}"
        )
    ctx = LoadContext(main_tabs=main_tabs, shared=shared, refs=refs)
    slots = merge_packer_dicts([unit.slots(ctx) for unit in LOAD_UNITS])
    expected_keys = load_slot_keys()
    got_keys = frozenset(slots)
    if got_keys != expected_keys:
        raise ValueError(
            "Load slot keys mismatch vs packer: "
            f"missing={sorted(expected_keys - got_keys)} extra={sorted(got_keys - expected_keys)}"
        )
    return slots


def resolve_upload_path(upload_obj) -> str | None:
    """Filesystem path from a Gradio File value (string or file-like)."""
    if upload_obj is None:
        return None
    if isinstance(upload_obj, str):
        return upload_obj
    name = getattr(upload_obj, "name", None)
    return name if name else None


def empty_load_outputs(banner: str = "", detail: str = "*No data*") -> dict[str, Any]:
    """Named empty/error outputs. Keys match `pack_load_outputs`."""
    if detail and detail != "*No data*":
        banner += (
            f"<p style='color:var(--ov-muted);font-size:13px;margin:4px 0 0;'>{html.escape(detail.strip('*'))}</p>"
        )
    return collect_pack(banner=banner)


def _pack_error(err: LoadError) -> dict[str, Any]:
    if err.code == "not_found":
        return empty_load_outputs(detail=f"*{err.message}*")
    if err.code == "mismatch":
        selected_key = err.selected or ""
        detected_key = err.detected or ""
        err_msg = (
            f"Format mismatch: selected "
            f"<b>{html.escape(FORMAT_LABELS.get(selected_key, selected_key))}</b>"
            f" but file detected as "
            f"<b>{html.escape(FORMAT_LABELS.get(detected_key, detected_key))}</b>."
        )
        return empty_load_outputs(
            banner=f"<p style='color:#dc2626;'>{err_msg}</p>",
            detail="*Please select the correct format and try again.*",
        )
    if err.code == "unknown":
        return empty_load_outputs(
            banner=f"<p style='color:#dc2626;'>{html.escape(err.message)}</p>",
            detail="*Unrecognized trajectory file.*",
        )
    return empty_load_outputs(
        banner=f"<p style='color:#dc2626;'>Error: {html.escape(err.message)}</p>",
        detail="*Error loading file.*",
    )


def pack_load_outputs(result: LoadedSession | LoadError, dark: bool = False) -> dict[str, Any]:
    """Map a load result onto named Gradio slots."""
    if isinstance(result, LoadError):
        return _pack_error(result)
    packed = collect_pack(result, dark=dark)
    expected = load_slot_keys()
    got = frozenset(packed)
    if got != expected:
        raise ValueError(f"Load packer keys mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    return packed


def do_load(upload_obj, dark=False, selected_format="", *, slots: dict[str, Any] | None = None):
    """Load wrapper: surface unexpected failures as a visible banner.

    When *slots* is provided (the live dashboard), return a Gradio component
    dict. Tests call ``pack_load_outputs`` / ``empty_load_outputs`` directly.
    """
    try:
        file_path = resolve_upload_path(upload_obj)
        result = load_session(file_path or "", format_hint=selected_format or "")
        packed = pack_load_outputs(result, dark=bool(dark))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        packed = empty_load_outputs(
            banner=(f"<p style='color:#dc2626;'>Error loading trajectory: {html.escape(str(exc))}</p>"),
            detail="*Failed to load — see console for details.*",
        )
    if slots is None:
        return packed
    return {slots[key]: packed[key] for key in slots}


def bind_load(*, file_upload, load_btn, format_selector, state_dark, slots: dict[str, Any]):
    """Wire Load Trajectory click and auto-load on file change. Returns both events."""
    expected = load_slot_keys()
    got = frozenset(slots)
    if got != expected:
        raise ValueError(
            f"Load slot keys mismatch vs packer: missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )

    def _do_load(upload_obj, dark=False, selected_format=""):
        return do_load(upload_obj, dark, selected_format, slots=slots)

    outputs = list(slots.values())
    _load_ev = load_btn.click(
        fn=_do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=outputs,
    )
    _upload_ev = file_upload.change(
        fn=_do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=outputs,
    )
    return _load_ev, _upload_ev
