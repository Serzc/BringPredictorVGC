"""main.py test no se corre dos veces: con la marca .test_run_done presente se niega (sin --force)."""
import pytest

import main


def test_test_run_refuses_second_time(tmp_path, monkeypatch):
    marca = tmp_path / ".test_run_done"
    marca.write_text("inicio 2026-01-01T00:00:00\nfin 2026-01-01T01:00:00\n", encoding="utf-8")
    monkeypatch.setattr(main, "TEST_DONE", marca)
    with pytest.raises(SystemExit, match="ya se corrio"):
        main.paso_test(force=False)
    assert marca.read_text(encoding="utf-8").count("inicio") == 1     # no se escribio nada nuevo
