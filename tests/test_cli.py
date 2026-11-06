import pytest

from carcounter import cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    configs, experiments = tmp_path / "configs", tmp_path / "experiments"
    configs.mkdir()
    (configs / "front-view.yaml").write_text("name: front-view\n")
    monkeypatch.setattr(cli, "CONFIGS", configs)
    monkeypatch.setattr(cli, "EXPERIMENTS", experiments)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    return tmp_path


def test_experiment_created_from_preset(project):
    d = cli.experiment_dir("front-view")
    assert (d / "experiment.yaml").read_text() == "name: front-view\n"
    assert (d / "videos").is_dir() and (d / "lines").is_dir()


def test_existing_experiment_without_folders_is_completed(project):
    d = project / "experiments" / "front-view"
    d.mkdir(parents=True)
    (d / "experiment.yaml").write_text("name: custom\n")
    assert cli.experiment_dir("front-view") == d
    assert (d / "experiment.yaml").read_text() == "name: custom\n"  # local copy is not overwritten
    assert (d / "videos").is_dir()


def test_unknown_experiment_exits(project):
    with pytest.raises(SystemExit):
        cli.experiment_dir("nope")


def test_no_videos_exits(project):
    d = cli.experiment_dir("front-view")
    with pytest.raises(SystemExit):
        cli.list_videos(d, None)
