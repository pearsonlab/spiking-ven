"""Download the R469 song corpus.

Replaces the shell + curl approach used previously. Two reasons this is Python and
stdlib-only:

* Both hosts sit behind CloudFront, which answers the default ``curl``/``urllib``
  User-Agent with ``403 Request blocked``. The browser UA is set here, in code, so the
  failure cannot come back through a forgotten shell flag.
* No ``requests`` dependency, and no dependence on ``curl``/``unzip`` being present.

Sources
-------
Koch 2024, adult zebra finch R469 -- WAV + Evsonganaly ``.not.mat`` annotations.
    https://doi.org/10.18738/T8/SAWMUN  (TDL Dataverse, file id 650237)

Duarte Ortiz et al. 2025 -- ``adult_songs/data.npz``.
    https://research.repository.duke.edu/record/438
    OPTIONAL: those are spectrogram features for the paper's rate-based Wilson-Cowan
    models, and are not used by this spiking pipeline. The endpoint is also flaky (it has
    returned an empty HTTP 202), so a failure here is reported and skipped, never fatal.
"""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["fetch_r469", "fetch_optional_duke_features", "BROWSER_UA"]

# CloudFront 403s the default urllib/curl agent on both hosts.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

KOCH_URL = "https://dataverse.tdl.org/api/access/datafile/650237"
DUKE_URL = (
    "https://research.repository.duke.edu/record/438/files/"
    "example_data_for_model_simulations.zip"
)


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, overridable for machines with a managed trust store.

    Set ``SVEN_CA_BUNDLE`` to a PEM bundle if your Python cannot verify these hosts --
    e.g. behind a TLS-inspecting corporate proxy, or with a conda/venv OpenSSL whose
    trust store is incomplete. If downloading remains impossible, skip it entirely and
    place the WAV + ``.not.mat`` pairs in ``<data-dir>/song_wavs`` by hand; every later
    stage reads from there and never needs the network.
    """
    bundle = os.environ.get("SVEN_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()


def _download(url: str, dest: Path, *, timeout: int = 120) -> None:
    """Stream ``url`` to ``dest``, presenting a browser User-Agent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            with open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
    except urllib.error.URLError as exc:
        # urllib wraps the TLS failure in URLError, so inspect .reason rather than
        # trying to catch ssl.SSLCertVerificationError directly.
        if not isinstance(getattr(exc, "reason", None), ssl.SSLError):
            raise
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"TLS verification failed for {url}.\n"
            "This Python cannot verify the host's certificate chain. Either set "
            "SVEN_CA_BUNDLE to a PEM bundle that includes the public roots, or skip the "
            "download and place the WAV + .not.mat pairs in <data-dir>/song_wavs "
            "manually -- no later stage needs the network."
        ) from exc
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{url} returned an empty body")


def fetch_r469(data_dir: Path, *, verbose: bool = True) -> Path:
    """Download and flatten the R469 WAV + annotation pairs into ``data_dir/song_wavs``.

    Returns the ``song_wavs`` directory. Skips the download if it already has WAVs.
    """
    wav_dir = Path(data_dir) / "song_wavs"
    if wav_dir.is_dir() and any(wav_dir.glob("*.wav")):
        if verbose:
            n = len(list(wav_dir.glob("*.wav")))
            print(f"  song_wavs/ already present ({n} WAVs), skipping download")
        return wav_dir

    zip_path = Path(data_dir) / "R469.zip"
    if verbose:
        print("  downloading R469.zip (~10 MB) from TDL Dataverse ...")
    _download(KOCH_URL, zip_path)

    wav_dir.mkdir(parents=True, exist_ok=True)
    # The archive nests everything under a top-level R469/ directory; flatten it.
    with zipfile.ZipFile(zip_path) as zf:
        wanted = [n for n in zf.namelist() if n.endswith((".wav", ".not.mat"))]
        for name in wanted:
            target = wav_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink(missing_ok=True)

    n_wav = len(list(wav_dir.glob("*.wav")))
    n_ann = len(list(wav_dir.glob("*.not.mat")))
    if n_wav == 0:
        raise RuntimeError(f"no WAV files extracted into {wav_dir}")
    if verbose:
        print(f"  -> {wav_dir} ({n_wav} WAVs, {n_ann} annotation files)")
    return wav_dir


def fetch_optional_duke_features(data_dir: Path, *, verbose: bool = True) -> Path | None:
    """Best-effort fetch of the Duke feature archive. Returns None if unavailable.

    Never raises: this archive is not used by the spiking pipeline.
    """
    out_dir = Path(data_dir) / "example_data_for_model_simulations"
    if (out_dir / "adult_songs" / "data.npz").exists():
        if verbose:
            print("  Duke features already present, skipping")
        return out_dir

    zip_path = Path(data_dir) / "example_data_for_model_simulations.zip"
    try:
        if verbose:
            print("  downloading optional Duke feature archive (~40 MB) ...")
        _download(DUKE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        if verbose:
            print(f"  -> {out_dir}")
        return out_dir
    except Exception as exc:  # noqa: BLE001 - optional, must never be fatal
        if verbose:
            print(f"  !! skipped optional Duke archive ({type(exc).__name__}: {exc})")
            print("     Not needed by the spiking pipeline; continuing.")
        return None
    finally:
        zip_path.unlink(missing_ok=True)
