from asrs_copilot.data import load_reports


def test_load_reports_drops_missing(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("id,narrative\n1,hello\n2,\n")
    df = load_reports(p)
    assert list(df["id"]) == [1]
