# ── main.py ─────────────────────────────────────────────────────
from __future__ import annotations

import os, pickle, time
from io import BytesIO
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import uvicorn
import numpy as np
import zipfile

from pipeline.pipeline import Pipeline
from pipeline.segmentation_filter import SegmentationFilter
from pipeline.user_palette_classification_filter import UserPaletteClassificationFilter
from palette_classification.palette import PaletteRGB
from utils.utils import grey_world                                   # AWB

# ╭──────────── helper ───────────────────────────────────────────╮
def load_palettes(dir_: str, *, pkl: bool = False) -> List[PaletteRGB]:
    acc: List[PaletteRGB] = []
    for fname in os.listdir(dir_):
        fpath = os.path.join(dir_, fname)
        if pkl and fname.endswith(".pkl"):
            acc.append(pickle.load(open(fpath, "rb")))
        elif (not pkl) and fname.endswith(".csv"):
            acc.append(PaletteRGB().load(fpath, header=True))
    return acc


def season_detail(base: str, sivc: List[int] | None) -> str:
    if sivc is None: return base.capitalize()
    s, i, v, c = sivc
    match base:
        case "spring":
            return ("Bright Spring" if i and v else
                    "Light Spring"  if v and not i else
                    "Warm Spring"   if i and not v else "Spring")
        case "summer":
            return ("Light Summer"  if v and not i else
                    "Soft Summer"   if not i and not v else
                    "Cool Summer"   if i and not v else "Summer")
        case "autumn":
            return ("Warm Autumn"   if i and not v else
                    "Soft Autumn"   if not i and not v else
                    "Deep Autumn"   if i and v else "Autumn")
        case "winter":
            return ("Bright Winter" if i and v else
                    "Cool Winter"   if v and not i else
                    "Deep Winter"   if i and not v else "Winter")
        case _:
            return base.capitalize()


def fmt_response(pdict: Dict) -> Dict:
    """Converte il dict restituito dal filtro in JSON friendly."""
    pal:   PaletteRGB = pdict["user_palette"]
    sivc:  List[int]  = pdict["metrics_vector"]
    return {
        "season":         season_detail(pal.description(), sivc),
        "base_season":    pal.description(),
        "metrics_vector": sivc,
        "dominants_rgb":  pdict["dominants_rgb"],               # {skin,hair,lips,eyes}
        "colors":         pal.colors().squeeze().permute(1, 0).tolist()
    }
# ╰───────────────────────────────────────────────────────────────╯


# ╭──────────── init FastAPI ─────────────────────────────────────╮
PALETTE_DIR          = "palette_classification/palettes"
reference_palettes   = load_palettes(PALETTE_DIR, pkl=False)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# mount /debug → /tmp  (immagini di debug)
app.mount("/debug", StaticFiles(directory="/tmp"), name="debug")

# pipeline
pipe = Pipeline()
pipe.add_filter(SegmentationFilter(model="local"))
pipe.add_filter(UserPaletteClassificationFilter(reference_palettes))
# ╰───────────────────────────────────────────────────────────────╯


# ────────────────────────────────────────────────────────────────
# POST /analyze          → output JSON pulito
# POST /analyze_debug    → output JSON + lista PNG in /tmp
# POST /analyze_masks    → ZIP con le 4 maschere binarie
# ────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    out = pipe.execute(image)              # ← now a dict with many keys

    # ------------------------------------------------------------------
    # unwrap the palette object only to extract `base_season`
    user_pal = out["user_palette"]
    base = user_pal.description()

    # recompute (or keep) friendly sub-season
    detailed = season_detail(base, out["metrics_vector"])

    # build final response: just inject the new dicts
    return {
        "season":          detailed,
        "base_season":     base,
        "metrics_vector":  out["metrics_vector"],
        "metrics_raw":     out["metrics_raw"],
        "thresholds":      out["thresholds"],
        "dominants_rgb":   out["dominants_rgb"],
        "pixel_counts":    out["pixel_counts"],
        "grey_world_scale":out["grey_world_scale"],
        "model_version":   out["model_version"],
        "colors":          out["user_palette"].colors()
                               .squeeze().permute(1,0).tolist()
    }


@app.post("/analyze_debug")
async def analyze_debug(file: UploadFile = File(...)):
    before = {f for f in os.listdir("/tmp") if f.endswith(".png")}

    img_pil = Image.open(BytesIO(await file.read())).convert("RGB")
    img_eq  = Image.fromarray(grey_world(np.array(img_pil)))

    out = pipe.execute(img_eq, verbose=True)   # PNG scritti in /tmp
    time.sleep(0.2)                            # sync I/O

    after = {f for f in os.listdir("/tmp") if f.endswith(".png")}
    debug_pngs = sorted(after - before)
    resp = fmt_response(out)
    resp["debug_images"] = [f"/debug/{f}" for f in debug_pngs]
    return resp


@app.post("/analyze_masks", response_class=FileResponse,
          responses={200: {"content": {"application/zip": {}}}})
async def analyze_masks(file: UploadFile = File(...)):
    img_pil = Image.open(BytesIO(await file.read())).convert("RGB")
    img_eq  = Image.fromarray(grey_world(np.array(img_pil)))

    out = pipe.execute(img_eq)                      # ritorna anche filtered_masks
    paths = []
    for name, mask in zip(["skin", "hair", "lips", "eyes"], out["filtered_masks"]):
        arr  = (mask.cpu().numpy() * 255).astype("uint8")
        pth  = f"/tmp/{name}.png"
        Image.fromarray(arr).save(pth)
        paths.append(pth)

    zip_path = "/tmp/masks.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))

    return FileResponse(zip_path, media_type="application/zip",
                        filename="masks.zip")
# ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
