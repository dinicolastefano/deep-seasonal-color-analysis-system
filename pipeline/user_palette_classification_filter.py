# ── pipeline/user_palette_classification_filter.py ──────────────
from __future__ import annotations

"""
User-palette filter – extended output.

Restituisce:
• dominants_rgb, metrics_vector
• metrics_raw        (valori continui S I V C)
• pixel_counts       (# pixel post-filtro per regione)
• grey_world_scale   (fattori R,G,B)
• model_version
"""

import sys, cv2, numpy as np, torch, logging
from os import path
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from .abstract_filter import AbstractFilter
from utils import utils, segmentation_labels
from palette_classification import palette
from sklearn.cluster import KMeans
from utils.utils import grey_world                      # bilanciamento veloce

MODEL_VERSION = "dscas-1.3.2"

# ───────── logger ───────────────────────────────────────────────
log = logging.getLogger("palette_filter")
log.setLevel(logging.INFO)

# ───────── optional config YAML ─────────────────────────────────
try:
    from utils.config import CONFIG
except (ModuleNotFoundError, FileNotFoundError):
    CONFIG = {}

# ╭──────── dataclass parametri regione ─────────────────────────╮
@dataclass
class RegionCfg:
    kernel_open: int
    kmeans_k:    int
    pick:        str
    crop_radius: int = 10     # (solo eyes)

    @classmethod
    def from_dict(cls, d: Dict) -> "RegionCfg":
        return cls(
            kernel_open=int(d.get("kernel_open", 5)),
            kmeans_k   =int(d.get("kmeans_k",    3)),
            pick       =     d.get("pick",       "bright"),
            crop_radius=int(d.get("crop_radius", 10)),
        )
# ╰──────────────────────────────────────────────────────────────╯


# ───────── funzione colore dominante con filtro outlier IQR ────
def _dominant(
    pix: np.ndarray,
    k: int,
    strategy: str,
    hair_rgb: np.ndarray | None = None,
    skin_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    """Ritorna il colore dominante; None se troppo pochi pixel."""

    if pix.shape[0] < k:
        return None

    # OUTLIER su luminosità (Value) con IQR ----------------------
    pix_raw = pix.copy()
    v = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)\
          .reshape(-1, 3)[:, 2]
    q1, q3 = np.percentile(v, [25, 75])
    iqr    = q3 - q1
    good   = (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
    pix    = pix[good] if good.any() else pix_raw
    # ------------------------------------------------------------

    km  = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pix)
    ctr = km.cluster_centers_.astype(np.uint8)  # (k,3)

    match strategy:
        case "bright":
            labs = cv2.cvtColor(ctr[None], cv2.COLOR_RGB2LAB)[0]
            idx  = labs[:, 0].argmax()
        case "saturated":
            labs = cv2.cvtColor(ctr[None], cv2.COLOR_RGB2LAB)[0]
            idx  = np.sqrt(labs[:, 1]**2 + labs[:, 2]**2).argmax()
        case "largest":
            idx  = np.bincount(km.labels_).argmax()
        case "eyes":
            hsv = cv2.cvtColor(ctr[None], cv2.COLOR_RGB2HSV)[0][:, 0]
            target = []
            if hair_rgb is not None:
                target.append(cv2.cvtColor(hair_rgb[None,None],
                                           cv2.COLOR_RGB2HSV)[0,0,0])
            if skin_rgb is not None:
                target.append(cv2.cvtColor(skin_rgb[None,None],
                                           cv2.COLOR_RGB2HSV)[0,0,0])
            if target:
                idx = np.abs(hsv[:,None] - np.array(target)[None,:]).min(axis=1).argmax()
            else:
                idx = hsv.argmax()
        case _:
            idx = 0
    return ctr[idx]


# ───────────── filtro principale ────────────────────────────────
class UserPaletteClassificationFilter(AbstractFilter):
    """Estrae 4 dominanti, calcola S-I-V-C, output esteso."""

    thresholds: Tuple[float, float, float] = (0.20, 0.55, 0.55)   # C, I, V

    def __init__(self, reference_palettes):
        cfg_blk = CONFIG.get("sampling", {})
        default = {"kernel_open": 5, "kmeans_k": 3, "pick": "bright",
                   "crop_radius": 10}
        self.cfg: Dict[str, RegionCfg] = {
            n: RegionCfg.from_dict(cfg_blk.get(n, default))
            for n in ["skin", "hair", "lips", "eyes"]
        }

        self.reference = reference_palettes
        rel = ["skin", "hair", "lips", "eyes"]
        self.rel_idx = [utils.from_key_to_index(segmentation_labels.labels, l)
                        for l in rel]

    # --- tipi I/O ---
    def input_type(self):
        return tuple
    def output_type(self):
        return dict

    # ------------------------------------------------------------
    def execute(self, inp, device=None, verbose: bool = False):
        img, masks = inp                                    # img uint8 (3,H,W)

        # grey-world + scale
        means  = img.float().mean(dim=(1,2))
        scale  = means.mean() / means
        img    = grey_world(img)
        gw_scale = scale.tolist()

        masks = masks[self.rel_idx]                         # (4,H,W)
        dominants: List[torch.Tensor] = []
        pixel_counts: Dict[str,int]  = {}

        skin_rgb: np.ndarray | None = None
        hair_rgb: np.ndarray | None = None

        for j, name in enumerate(["skin", "hair", "lips", "eyes"]):
            cfg = self.cfg[name]
            kernel = np.ones((cfg.kernel_open, cfg.kernel_open), np.uint8)
            mask   = cv2.morphologyEx(masks[j].byte().numpy(),
                                      cv2.MORPH_OPEN, kernel).astype(bool)

            # Crop pupilla per “eyes”
            if name == "eyes" and cfg.crop_radius > 0:
                ys, xs = np.where(mask)
                if xs.size and ys.size:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    r, h, w = cfg.crop_radius, *mask.shape
                    small = np.zeros_like(mask, dtype=bool)
                    x0,x1 = max(cx-r,0), min(cx+r,w)
                    y0,y1 = max(cy-r,0), min(cy+r,h)
                    small[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
                    mask = small

            if not mask.any():
                dom = np.array([0,0,0], np.uint8)
                dominants.append(torch.tensor(dom).reshape(3,1,1))
                pixel_counts[name] = 0
                continue

            pix_raw = img[:, mask].permute(1,0).numpy()     # (N,3)

            # ---------- HSV percentile filter (auto-adaptivo) ----
            hsv = cv2.cvtColor(pix_raw.reshape(-1,1,3),
                               cv2.COLOR_RGB2HSV).reshape(-1,3)
            s_low, s_high = np.percentile(hsv[:,1], [10,90])
            v_low, v_high = np.percentile(hsv[:,2], [10,90])
            good = ((hsv[:,1] >= s_low) & (hsv[:,1] <= s_high) &
                    (hsv[:,2] >= v_low) & (hsv[:,2] <= v_high))
            pix  = pix_raw[good] if good.any() else pix_raw
            # ------------------------------------------------------
            pixel_counts[name] = pix.shape[0]

            dom = _dominant(
                pix, cfg.kmeans_k, cfg.pick,
                hair_rgb=hair_rgb if name=="eyes" else None,
                skin_rgb=skin_rgb if name=="eyes" else None)
            if dom is None:
                dom = pix.mean(axis=0).astype(np.uint8)

            dominants.append(torch.tensor(dom).reshape(3,1,1))
            if name=="skin": skin_rgb = dom
            if name=="hair": hair_rgb = dom

        dominants_t = torch.stack(dominants).type(torch.uint8)
        skin, hair, lips, eyes = dominants_t

        # ----- metriche raw --------------------------------------
        subtone_val   = palette.compute_subtone(lips)
        intensity_val = palette.compute_intensity(skin)
        value_val     = palette.compute_value(skin, hair, eyes)
        contrast_val  = palette.compute_contrast(hair, eyes)

        pal_dom = palette.PaletteRGB("dominants", dominants_t)
        pal_dom.compute_metrics_vector(
            subtone_val, intensity_val, value_val,
            contrast_val, self.thresholds)
        user_pal = palette.classify_user_palette(
            pal_dom, self.reference, True)
        # ----------------------------------------------------------

        return {
            "user_palette":  user_pal,
            "dominants_rgb": {
                "skin": skin.reshape(3).tolist(),
                "hair": hair.reshape(3).tolist(),
                "lips": lips.reshape(3).tolist(),
                "eyes": eyes.reshape(3).tolist(),
            },
            "metrics_vector": pal_dom.metrics_vector().tolist(),
            "metrics_raw": {
                "subtone":   1 if subtone_val == "warm" else 0,
                "intensity": round(float(intensity_val), 3),
                "value":     round(float(value_val), 3),
                "contrast":  None if contrast_val is None
                             else round(float(contrast_val), 3),
            },
            "thresholds": {
                "contrast":  self.thresholds[0],
                "intensity": self.thresholds[1],
                "value":     self.thresholds[2],
            },
            "pixel_counts":     pixel_counts,
            "grey_world_scale": [round(float(s),3) for s in gw_scale],
            "model_version":    MODEL_VERSION,
        }
