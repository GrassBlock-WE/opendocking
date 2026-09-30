# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetching structures from the RCSB Protein Data Bank.

The two endpoints used here are the ones the RCSB documents for programmatic
access:

* ``https://files.rcsb.org/download/<PDB_ID>.pdb`` -- the PDB-format coordinate
  file of an entry;
* ``https://files.rcsb.org/ligands/download/<CCD_ID>.sdf`` -- the SDF of a
  chemical component (``<CCD_ID>_ideal.sdf`` and ``_model.sdf`` variants are
  accepted by the same endpoint and by :func:`fetch_ligand_sdf`).

Everything is plain :mod:`urllib`, so no HTTP library has to be installed, and
every failure mode is turned into a :class:`RuntimeError` whose message says
what went wrong: a caller of the CLI should never see a raw ``URLError``
traceback because a laptop was offline.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "DEFAULT_TIMEOUT",
    "RCSB_LIGAND_URL",
    "RCSB_PDB_URL",
    "USER_AGENT",
    "fetch_ligand_sdf",
    "fetch_pdb",
    "validate_ligand_id",
    "validate_pdb_id",
]

PathLike = Union[str, os.PathLike]

#: The PDB-format coordinate file of an entry.
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

#: The SDF of a chemical component.
RCSB_LIGAND_URL = "https://files.rcsb.org/ligands/download/{ligand_id}.sdf"

#: Seconds before a stalled connection is abandoned.
DEFAULT_TIMEOUT = 30.0

#: RCSB asks automated clients to identify themselves.
USER_AGENT = "OpenDocking/0.1 (+https://github.com/opendocking/odock)"


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------


def validate_pdb_id(pdb_id: str) -> str:
    """Validate a PDB entry identifier and return it upper-cased.

    A PDB ID is exactly four alphanumeric characters (``"3PTB"``, ``"1M17"``,
    and the extended five-character form is not served by
    ``files.rcsb.org/download``).  A blank or malformed identifier raises
    :class:`ValueError` *before* any request is made, so a typo cannot turn
    into a confusing 404.
    """
    if not isinstance(pdb_id, str):
        raise ValueError(f"a PDB ID must be a string, got {type(pdb_id).__name__}")
    value = pdb_id.strip().upper()
    if len(value) != 4:
        raise ValueError(
            f"a PDB ID is four characters, got {pdb_id!r} ({len(value)})"
        )
    if not value.isalnum() or not value.isascii():
        raise ValueError(f"a PDB ID must be alphanumeric, got {pdb_id!r}")
    return value


def validate_ligand_id(ligand_id: str) -> str:
    """Validate an RCSB chemical-component identifier and upper-case it.

    Component IDs (the three-letter codes in a PDB ``HET`` record: ``"BEN"``,
    ``"ATP"``, ``"STI"``) are one to three alphanumeric characters.  The
    documented RCSB variants ``"<ID>_ideal"`` and ``"<ID>_model"`` are accepted
    as well, because ``files.rcsb.org/ligands/download`` serves them from the
    same path.
    """
    if not isinstance(ligand_id, str):
        raise ValueError(
            f"a ligand ID must be a string, got {type(ligand_id).__name__}"
        )
    value = ligand_id.strip().upper()
    base, _, variant = value.partition("_")
    if variant and variant not in ("IDEAL", "MODEL"):
        raise ValueError(
            f"unknown ligand file variant {variant!r} in {ligand_id!r}; "
            "expected an optional '_ideal' or '_model'"
        )
    if not 1 <= len(base) <= 3:
        raise ValueError(
            f"an RCSB chemical-component ID is one to three characters, got {ligand_id!r}"
        )
    if not base.isalnum() or not base.isascii():
        raise ValueError(f"a ligand ID must be alphanumeric, got {ligand_id!r}")
    return value


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _download(url: str, what: str, timeout: float) -> str:
    """GET `url`, translating every transport failure into a `RuntimeError`."""
    if timeout is not None and float(timeout) <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"{what} was not found at {url} (HTTP 404); check the identifier"
            ) from exc
        raise RuntimeError(f"{what} could not be downloaded: HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"{what} could not be downloaded from {url}: {exc.reason}"
        ) from exc
    except (TimeoutError, OSError) as exc:  # socket timeouts, DNS, resets
        raise RuntimeError(f"{what} could not be downloaded from {url}: {exc}") from exc

    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="replace")
    else:  # pragma: no cover - urlopen always yields bytes
        text = str(payload)
    if not text.strip():
        raise RuntimeError(f"{what} downloaded from {url} is empty")
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_pdb(
    pdb_id: str, out: Optional[PathLike] = None, *, timeout: float = DEFAULT_TIMEOUT
) -> str:
    """Download a PDB entry and save it, returning its text.

    Parameters
    ----------
    pdb_id
        Four-character entry identifier, case-insensitive (``"3ptb"``).
    out
        Where to write the file.  Defaults to ``<ID>.pdb`` in the working
        directory, upper-cased.
    timeout
        Connection/read timeout in seconds (default 30).

    Returns
    -------
    The file text, which has also been written to disk.

    Raises
    ------
    ValueError
        The identifier is not four alphanumeric characters.
    RuntimeError
        The server answered 404, refused the connection, or timed out.
    """
    identifier = validate_pdb_id(pdb_id)
    path = Path(out) if out is not None else Path(f"{identifier}.pdb")
    text = _download(
        RCSB_PDB_URL.format(pdb_id=identifier),
        f"PDB entry {identifier}",
        timeout,
    )
    _write(path, text)
    return text


def fetch_ligand_sdf(
    ligand_id: str,
    out: Optional[PathLike] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Download a chemical component's SDF and save it, returning its text.

    Parameters
    ----------
    ligand_id
        One to three alphanumeric characters (``"BEN"``), optionally with the
        RCSB file variants ``"_ideal"`` or ``"_model"``.
    out
        Where to write the file.  Defaults to ``<ID>.sdf`` in the working
        directory, upper-cased.
    timeout
        Connection/read timeout in seconds (default 30).

    Returns
    -------
    The file text, which has also been written to disk.

    Raises
    ------
    ValueError
        The identifier is malformed.
    RuntimeError
        The server answered 404, refused the connection, or timed out.
    """
    identifier = validate_ligand_id(ligand_id)
    path = Path(out) if out is not None else Path(f"{identifier}.sdf")
    text = _download(
        RCSB_LIGAND_URL.format(ligand_id=identifier),
        f"ligand {identifier}",
        timeout,
    )
    _write(path, text)
    return text
