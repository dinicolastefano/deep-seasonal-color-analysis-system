# utils/config.py
# ────────────────────────────────────────────────────────────────
"""Loader della configurazione YAML.

✓ Cerca per default `config.yaml` nella root del progetto.
✓ Se definisci lʼenv VAR  COLORCFG  usa quel percorso.
✓ Solleva FileNotFoundError se il file non esiste.
✓ Espone il dict globale CONFIG, pronto da importare.
"""

from __future__ import annotations
import os, yaml, logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Percorso di default: cartella root (una directory sopra 'utils/')
_DEFAULT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yaml")
)

def load_config(path: str | None = None) -> dict:
    """Ritorna il dict con tutta la configurazione."""
    cfg_path = (
        path
        or os.environ.get("COLORCFG")   # 1) variabile d’ambiente
        or _DEFAULT_PATH                # 2) fallback
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    log.info(f"Loaded config: {cfg_path}")
    return data

# oggetto globale per import diretto
CONFIG: dict = load_config()
