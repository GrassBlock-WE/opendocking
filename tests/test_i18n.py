# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the workbench translation layer (:mod:`odock.gui.i18n`).

The important properties are structural rather than cosmetic: the two
dictionaries must stay complete and in sync, ``tr`` must fail loudly on a key
that does not exist (a silent English fallback would hide a missing
translation from every test), and the English labels must stay short enough to
fit the inspector and the menus.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from odock.gui import i18n

GUI_DIR = Path(__file__).resolve().parent.parent / "python" / "odock" / "gui"

#: Labels fixed by the specification, as ``key: (english, 中文)``.
MANDATED_LABELS = {
    "action.hetero": ("Ions & cofactors…", "离子与辅酶…"),
    "action.protonation": ("Protonation…", "质子化…"),
    "action.assign_charges": ("Assign charges", "赋电荷"),
    "action.strip_ligand": ("Strip ligand", "剥离配体"),
    "action.write_flex": ("Write flexible PDBQT…", "导出柔性 PDBQT…"),
    "action.ligand_charges": ("Charges + merge H", "电荷与并氢"),
    "action.lock_bond": ("Lock a bond", "锁定化学键"),
    "action.export_xlsx": ("Results XLSX…", "结果 XLSX…"),
    "action.flexible_residues": ("Add flexible residue…", "添加柔性残基…"),
    "action.show_interactions": ("Show interactions", "显示相互作用"),
    "action.detect_pockets": ("Detect pockets…", "探测口袋…"),
    "action.grid_info": ("Grid size info", "网格信息"),
    "action.play_poses": ("Play poses", "播放构象"),
    "menu.protein_style": ("Protein style", "蛋白样式"),
    "menu.ligand_style": ("Ligand style", "配体样式"),
    "menu.panels": ("Panels", "面板"),
}

#: Actions that never open a dialog, so their label must not end in an ellipsis.
NO_DIALOG_ACTIONS = {
    "action.remove_waters",
    "action.strip_ligand",
    "action.assign_charges",
    "action.detect_torsions",
    "action.ligand_charges",
    "action.lock_bond",
    "action.fit_box",
    "action.centre_box",
    "action.show_box",
    "action.grid_info",
    "action.use_vina",
    "action.use_vinardo",
    "action.use_ad4",
    "action.start",
    "action.pause_resume",
    "action.abort",
    "action.score",
    "action.engine_settings",
    "action.show_interactions",
    "action.clear_annotations",
    "action.play_poses",
    "action.axes",
    "action.ssao",
    "action.measure",
    "action.clear_measurements",
    "action.reset_view",
}


@pytest.fixture(autouse=True)
def english():
    """Start and finish every test in English, whatever the host locale is."""
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def _fields(text: str):
    return sorted({name for _, name, _, _ in string.Formatter().parse(text) if name})


# ---------------------------------------------------------------------------
# the dictionaries
# ---------------------------------------------------------------------------


def test_english_and_chinese_are_in_sync():
    assert set(i18n.EN) == set(i18n.ZH)


def test_every_translation_is_a_non_empty_string():
    for name, table in (("EN", i18n.EN), ("ZH", i18n.ZH)):
        for key, value in table.items():
            assert isinstance(value, str), f"{name}[{key!r}] is not a string"
            assert value.strip(), f"{name}[{key!r}] is empty"


def test_placeholders_match_between_the_two_languages():
    for key in sorted(i18n.EN):
        assert _fields(i18n.EN[key]) == _fields(i18n.ZH[key]), key


def test_available_keys_agrees_with_both_tables():
    keys = i18n.available_keys()
    assert keys == set(i18n.EN) == set(i18n.ZH)
    assert "menu.file" in keys
    assert "log.receptor" in keys


def test_the_chinese_menu_titles_are_translated():
    """A light smell test: no menu title is left in ASCII."""
    for key, english in i18n.EN.items():
        if not key.startswith(("menu.", "tab.", "dock.", "group.")):
            continue
        chinese = i18n.ZH[key]
        assert not chinese.isascii(), f"{key} was not translated ({english!r})"


def test_top_level_accelerators_are_unique_per_language():
    for name in ("EN", "ZH"):
        table = getattr(i18n, name)
        letters = [
            value.split("&")[1][0].lower()
            for key, value in table.items()
            if key.startswith("menu.") and "&" in value
        ]
        # The seven menus of the menu bar plus the File ▸ Export submenu.
        assert len(letters) == 8, f"{name} should have eight accelerated menus"
        assert len(letters) == len(set(letters)), f"{name}: {letters}"


@pytest.mark.parametrize("key,expected", sorted(MANDATED_LABELS.items()))
def test_mandated_labels(key, expected):
    english, chinese = expected
    assert i18n.EN[key] == english
    assert i18n.ZH[key] == chinese


def test_english_labels_are_short():
    for key, value in i18n.EN.items():
        if key.startswith("action."):
            limit = 22
        elif key.startswith("btn."):
            limit = 18
        else:
            continue
        label = value.replace("&", "").rstrip("…")
        assert len(label) <= limit, f"{key} = {value!r} is {len(label)} characters"


def test_actions_without_a_dialog_have_no_ellipsis():
    for key in sorted(NO_DIALOG_ACTIONS):
        assert not i18n.EN[key].endswith("…"), key
        assert not i18n.ZH[key].endswith("…"), key


def test_keys_built_at_runtime_exist():
    for axis in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
        assert f"grid.{axis}" in i18n.EN
    assert "grid.spacing" in i18n.EN
    for kind in ("cofactor", "ion", "ligand", "other", "solvent"):
        assert f"kind.{kind}" in i18n.EN
    for role in ("receptor", "ligand"):
        assert f"style.role.{role}" in i18n.EN
    for what in (
        "receptor_pdbqt",
        "ligand_pdbqt",
        "poses_pdbqt",
        "cleaned_pdb",
        "native_ligand",
        "gpf",
        "dpf",
        "config",
        "xlsx",
        "csv",
        "svg",
    ):
        assert f"export.{what}" in i18n.EN
    for kind in ("hbond", "salt_bridge", "pi_pi", "cation_pi", "hydrophobic", "clash"):
        assert f"interaction.{kind}" in i18n.EN


def test_every_literal_key_used_by_the_gui_exists():
    """A ``tr("...")`` typo must fail here rather than at the first click."""
    pattern = re.compile(r'tr\(\s*"([A-Za-z0-9_.]+)"\s*[,)]')
    for name in ("app.py", "dialogs.py"):
        text = (GUI_DIR / name).read_text(encoding="utf-8")
        keys = set(pattern.findall(text))
        assert keys, f"no literal tr() keys found in {name}"
        missing = sorted(key for key in keys if key not in i18n.EN)
        assert not missing, f"{name} uses unknown keys: {missing}"


def test_no_visible_literal_is_left_in_the_gui_sources():
    """A crude guard against a label that bypassed ``tr`` entirely."""
    forbidden = (
        "Key interacting residues",
        "Poses and monitoring",
        "Estimate the affinity maps",
        "Write flexible-receptor PDBQT",
        "Strip co-crystallised ligand",
        "Assign charges and AD4 types",
    )
    for name in ("app.py", "dialogs.py"):
        text = (GUI_DIR / name).read_text(encoding="utf-8")
        for literal in forbidden:
            assert literal not in text, f"{name} still contains {literal!r}"


# ---------------------------------------------------------------------------
# the functions
# ---------------------------------------------------------------------------


def test_languages_cover_both_tables():
    assert set(i18n.LANGUAGES) == {"en", "zh"}
    for code in i18n.LANGUAGES:
        i18n.set_language(code)
        assert i18n.current_language() == code
        assert i18n.language_name(code)


def test_set_language_round_trips_and_rejects_unknown_codes():
    i18n.set_language("zh")
    assert i18n.current_language() == "zh"
    i18n.set_language("en")
    assert i18n.current_language() == "en"
    with pytest.raises(ValueError):
        i18n.set_language("de")
    assert i18n.current_language() == "en"


def test_tr_raises_for_an_unknown_key_in_every_language():
    for code in ("en", "zh"):
        i18n.set_language(code)
        with pytest.raises(KeyError):
            i18n.tr("no.such.key")
        with pytest.raises(KeyError):
            i18n.tr_ctx("no.such.key")


def test_tr_formats_its_placeholders():
    i18n.set_language("en")
    assert (
        i18n.tr("log.receptor", n=1994, name="x.pdbqt")
        == "receptor: 1994 atoms from x.pdbqt"
    )
    assert i18n.tr("label.mode", index=2, total=6) == "mode 2 / 6"
    assert i18n.tr("label.box_info", volume=1234.0, npts=1000, mb=1.5).startswith(
        "1,234 Å³"
    )
    i18n.set_language("zh")
    assert "1994" in i18n.tr("log.receptor", n=1994, name="x.pdbqt")
    assert i18n.tr("label.mode", index=2, total=6) == "构象 2 / 6"


def test_a_missing_placeholder_argument_is_loud():
    with pytest.raises(KeyError):
        i18n.tr("log.receptor", n=1)


def test_tr_returns_the_other_language_after_a_switch():
    i18n.set_language("en")
    english = i18n.tr("action.start")
    i18n.set_language("zh")
    chinese = i18n.tr("action.start")
    assert english != chinese


def test_default_language_follows_the_system_locale(monkeypatch):
    monkeypatch.setattr(i18n.locale, "getdefaultlocale", lambda: ("zh_CN", "cp936"))
    assert i18n._default_language() == "zh"
    monkeypatch.setattr(i18n.locale, "getdefaultlocale", lambda: ("zh", None))
    assert i18n._default_language() == "zh"
    monkeypatch.setattr(i18n.locale, "getdefaultlocale", lambda: ("en_US", "UTF-8"))
    assert i18n._default_language() == "en"
    monkeypatch.setattr(i18n.locale, "getdefaultlocale", lambda: (None, None))
    assert i18n._default_language() == "en"

    def boom():
        raise OSError("no locale in this environment")

    monkeypatch.setattr(i18n.locale, "getdefaultlocale", boom)
    assert i18n._default_language() == "en"
