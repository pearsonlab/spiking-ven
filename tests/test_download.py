"""Corpus download and unpacking, with the network replaced by local fakes."""

import ssl
import urllib.error
import zipfile

import pytest

from spiking_ven.data import download


def _fake_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, f"contents of {name}")


def _serve(names, monkeypatch):
    """Make ``_download`` write a zip with ``names`` instead of fetching one."""
    calls = []

    def fake(url, dest, **kw):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _fake_zip(dest, names)

    monkeypatch.setattr(download, "_download", fake)
    return calls


def test_fetch_flattens_the_archive_and_keeps_only_song_files(tmp_path, monkeypatch):
    calls = _serve(["R469/a.wav", "R469/a.wav.not.mat", "R469/sub/b.wav",
                    "R469/readme.txt"], monkeypatch)
    wav_dir = download.fetch_r469(tmp_path, verbose=False)
    assert calls == [download.KOCH_URL]
    assert sorted(p.name for p in wav_dir.iterdir()) == ["a.wav", "a.wav.not.mat", "b.wav"]
    assert (wav_dir / "a.wav").read_text() == "contents of R469/a.wav"
    assert not (tmp_path / "R469.zip").exists(), "the archive is removed after unpacking"


def test_fetch_skips_when_wavs_are_already_present(tmp_path, monkeypatch):
    (tmp_path / "song_wavs").mkdir()
    (tmp_path / "song_wavs" / "x.wav").write_bytes(b"")
    calls = _serve(["R469/a.wav"], monkeypatch)
    download.fetch_r469(tmp_path, verbose=False)
    assert calls == []


def test_an_archive_without_wavs_is_an_error(tmp_path, monkeypatch):
    _serve(["R469/readme.txt"], monkeypatch)
    with pytest.raises(RuntimeError, match="no WAV"):
        download.fetch_r469(tmp_path, verbose=False)


def test_a_tls_failure_explains_the_workarounds(tmp_path, monkeypatch):
    def refuse(req, **kw):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("bad chain"))

    monkeypatch.setattr(download.urllib.request, "urlopen", refuse)
    dest = tmp_path / "x.zip"
    with pytest.raises(RuntimeError, match="SVEN_CA_BUNDLE"):
        download._download("https://example.invalid/x", dest)
    assert not dest.exists()


def test_other_network_errors_propagate(tmp_path, monkeypatch):
    def refuse(req, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(download.urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.URLError):
        download._download("https://example.invalid/x", tmp_path / "x.zip")


def test_an_empty_body_is_an_error(tmp_path, monkeypatch):
    class Empty:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b""

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda req, **kw: Empty())
    dest = tmp_path / "x.zip"
    with pytest.raises(RuntimeError, match="empty body"):
        download._download("https://example.invalid/x", dest)
    assert not dest.exists()


def test_requests_carry_the_browser_user_agent(tmp_path, monkeypatch):
    seen = {}

    def capture(req, **kw):
        seen["ua"] = req.get_header("User-agent")
        raise urllib.error.URLError("stop here")

    monkeypatch.setattr(download.urllib.request, "urlopen", capture)
    with pytest.raises(urllib.error.URLError):
        download._download("https://example.invalid/x", tmp_path / "x.zip")
    assert seen["ua"] == download.BROWSER_UA


def test_sven_ca_bundle_is_used_for_tls(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(download.ssl, "create_default_context",
                        lambda cafile=None: seen.setdefault("cafile", cafile))
    monkeypatch.setenv("SVEN_CA_BUNDLE", str(tmp_path / "roots.pem"))
    download._ssl_context()
    assert seen["cafile"] == str(tmp_path / "roots.pem")
