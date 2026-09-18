import json
import re

import pytest

from padel_analysis.analytics.rally import Rally, RallyContact
from padel_analysis.eval.contact_marks import pair_contacts
from padel_analysis.render.rally_page import RELIABILITY, page, rally_payload


def _rally():
    positions = {
        "near_1": {f: (0.0, -6.0) for f in range(91)},
        "far_1": {f: (1.0, 8.0) for f in range(91)},
    }
    contacts = (
        RallyContact(0, "raquette", None, "near_1"),
        RallyContact(15, "sol", (0.0, 6.0, 0.0), None),
        RallyContact(30, "raquette", None, "far_1"),
        RallyContact(45, "verre", (5.0, -8.0, 1.0), None),
    )
    return Rally(0, 90, 30.0, contacts, positions)


def _payload(truth=None):
    return rally_payload(_rally(), "r1", "Un echange", {"near_1": "Alice"}, "r1.mp4", truth)


def test_the_payload_is_plain_json():
    json.dumps(_payload())


def test_contacts_are_timed_from_the_rally_start():
    payload = _payload()
    assert [c["t"] for c in payload["contacts"]] == [0.0, 0.5, 1.0, 1.5]


def test_players_carry_their_name_or_their_slot():
    players = _payload()["players"]
    assert players["near_1"]["name"] == "Alice"
    assert players["far_1"]["name"] == "far_1"
    assert players["near_1"]["shots"] == 1


def test_a_player_path_is_thinned_to_one_point_every_three_frames():
    path = _payload()["players"]["near_1"]["path"]
    assert len(path) == 31
    assert path[1][0] == pytest.approx(0.1)


def test_the_summary_is_in_the_payload():
    assert _payload()["summary"] == {"duration": 3.0, "shots": 2, "walls": 1, "last": "verre"}


def test_no_truth_means_the_truth_mode_is_off():
    assert _payload()["truth"] is None


def test_the_truth_lines_keep_their_status():
    detected = {0: "raquette", 15: "sol", 30: "raquette", 45: "verre"}
    marks = {1: "raquette", 16: "sol", 60: "sol"}
    truth = _payload(pair_contacts(detected, marks))["truth"]
    assert [line["status"] for line in truth] == [
        "juste", "juste", "invente", "invente", "manque",
    ]


def test_the_page_embeds_every_rally_and_every_reliability_note():
    text = page([_payload(), {**_payload(), "id": "r2"}])
    data = json.loads(re.search(r"const DATA = (.*?);\n", text, re.DOTALL).group(1)
                      .replace("<\\/", "</"))
    assert [r["id"] for r in data["rallies"]] == ["r1", "r2"]
    assert data["reliability"] == RELIABILITY


def test_the_page_offers_a_toggle_for_every_panel():
    text = page([_payload()])
    for panel in ("frise", "plan", "chiffres", "frappes", "vitesse", "deplacements"):
        assert f'data-panel="{panel}"' in text


def test_a_name_cannot_close_the_data_script():
    payload = rally_payload(_rally(), "r1", "t", {"near_1": "</script><b>"}, "r1.mp4")
    assert "</script><b>" not in page([payload])
