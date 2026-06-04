"""Tests for deterministic action parsing (signal_processing.parse_action)."""

from sportagent.core.graph.signal_processing import parse_action

_CASES = [
    ("FINAL RECOMMENDATION: **BUY YES**", "BUY YES"),
    ("FINAL RECOMMENDATION: **BUY NO**", "BUY NO"),
    ("FINAL RECOMMENDATION: **HOLD**", "HOLD"),
    ("**Action:** BUY YES\nFINAL RECOMMENDATION: **BUY YES**", "BUY YES"),
    ("Action: HOLD", "HOLD"),
    ("we lean YES; Action: BUY YES here", "BUY YES"),
    ("", "HOLD"),
    ("no clear label, just prose about the knicks", "HOLD"),
]


def test_parse_action_cases():
    for text, want in _CASES:
        assert parse_action(text) == want, (text, want, parse_action(text))


if __name__ == "__main__":
    ok = True
    for text, want in _CASES:
        got = parse_action(text)
        status = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{status}: want={want!r} got={got!r}")
    print("ALL PASS" if ok else "SOME FAILED")