from types import SimpleNamespace

from echo_spin import provenance


def test_software_provenance_records_checkout_state(monkeypatch):
    """Capture versions and distinguish modified code from its Git revision."""
    monkeypatch.setattr(provenance, "version", lambda name: "test-version")
    monkeypatch.setattr(
        provenance.Path,
        "is_file",
        lambda path: True,
    )
    monkeypatch.setattr(
        provenance.Path,
        "exists",
        lambda path: True,
    )

    calls = []

    def run(command, **kwargs):
        """Return controlled Git results and retain the checkout location."""
        calls.append((command, kwargs["cwd"]))
        output = "abc123\n" if command[1] == "rev-parse" else " M README.md\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(provenance.subprocess, "run", run)

    report = provenance.software_provenance()

    assert report["packages"]["echo-spin"] == "test-version"
    assert report["git_revision"] == "abc123"
    assert report["git_dirty"] is True
    assert len(calls) == 2
    assert all(
        checkout == provenance.Path(provenance.__file__).resolve().parents[2]
        for _, checkout in calls
    )


def test_software_provenance_handles_unavailable_tools(monkeypatch):
    """Retain explicit unknown values when metadata or Git is unavailable."""
    def missing_version(name):
        """Represent an unavailable package installation."""
        raise provenance.PackageNotFoundError(name)

    def missing_git(*args, **kwargs):
        """Represent an unavailable Git executable."""
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance, "version", missing_version)
    monkeypatch.setattr(provenance.Path, "is_file", lambda path: True)
    monkeypatch.setattr(provenance.Path, "exists", lambda path: True)
    monkeypatch.setattr(provenance.subprocess, "run", missing_git)

    report = provenance.software_provenance()

    assert report["python"]
    assert all(value is None for value in report["packages"].values())
    assert report["git_revision"] is None
    assert report["git_dirty"] is None
