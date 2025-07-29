#!/usr/bin/env python
"""
Ray demo job: Render a sequence of Mandelbrot frames that zoom smoothly
towards what looks like a Juliabrot (I'm sure that's not techically correct, but you know).
"""

import os
from pathlib import Path
import matplotlib
import numpy as np
from PIL import Image
import ray
import secrets

from google.cloud import storage

# ----------------------- CONFIG ---------------------------------
N_FRAMES      = 32            # total frames (each frame spawns a separate task, which take around 6 minutes)
IMG_SIZE      = 1024          # square image (px)
MAX_ITER      = 3000          # iterations per pixel
CENTER        = (-0.743643887036, 0.131825904207)  # well-known Valley
START_SCALE   = 3.0           # width of view at frame 0
END_SCALE     = 6e-11         # width of view at final frame
# ----------------------------------------------------------------


OUT_DIR       = Path("frames")
BUCKET_NAME   = "heata-public-demo-assets"  # GCP bucket name for storing outputs


"""
Uploads a file to a GCP bucket using a service account key.
"""
def _upload_to_gcp(local_file: Path, dest_blob: str):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(str(local_file))




@ray.remote(num_cpus=1)
def render_frame(idx: int, job_secret_key: str) -> str:
    """
    Renders a single Mandelbrot frame and writes it to disk.
    idx ∈ [0, N_FRAMES - 1] determines the zoom level.
    hex_key is a unique id for this run.
    """
    # Smooth exponential zoom
    scale = START_SCALE * (END_SCALE / START_SCALE) ** (idx / (N_FRAMES - 1))
    xmin = CENTER[0] - scale / 2
    xmax = CENTER[0] + scale / 2
    ymin = CENTER[1] - scale / 2
    ymax = CENTER[1] + scale / 2

    # Pixel grid
    xs = np.linspace(xmin, xmax, IMG_SIZE, dtype=np.float64)
    ys = np.linspace(ymin, ymax, IMG_SIZE, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    iters = np.zeros(C.shape, dtype=np.uint16)

    # Escape-time algorithm
    mask = np.ones(C.shape, bool)
    for i in range(MAX_ITER):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = mask & (np.abs(Z) > 4.0)
        iters[escaped] = i
        mask[escaped] = False
        if not mask.any():
            break

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    mi = iters.max()

    if mi == 0:
        norm = iters
    else:
        norm = iters / mi                      # 0‥1

    cmap = matplotlib.colormaps["turbo"]   # or "brg", "inferno", "magma", …

    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)  # strip α, scale
    Image.fromarray(rgb).save(
        OUT_DIR / f"frame_{idx:04d}.png"
    )
    # Upload to GCP bucket heata-example-store/outputs using service account key
    dest_blob = f"outputs/{job_secret_key}/frame_{idx:04d}.png"
    local_file = OUT_DIR / f"frame_{idx:04d}.png"

    _upload_to_gcp(local_file, dest_blob)

    # remove the local file after upload
    os.remove(local_file)

    return f"{dest_blob} written"






"""
Download all frames from GCP bucket to local directory.
This function is useful for downloading all frames after the job is done.
Usage infomation is printed at the end of the main function.
"""
def download_all_from_gcp(job_secret_key: str):
    """
    Downloads all frames from the GCP bucket to the local directory.
    """
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for i in range(N_FRAMES):
        blob_name = f"outputs/{job_secret_key}/frame_{i:04d}.png"
        local_file = OUT_DIR / Path(blob_name).name
        print(f"Downloading {blob_name} to {local_file}")
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_file))

    print(f"All frames downloaded to {OUT_DIR}.")




def main():
    ray.init(address="auto")

    # Generate a unique secret key for this job, this is used to create a unique folder in GCP bucket which others cannot find
    job_secret_key = secrets.token_hex(16)

    print()
    print(f"To download all frames after the job is done, run: python -c \"import mandelbrot_zoom as m; m.download_all_from_gcp('{job_secret_key}')\"")


    # Render frames in parallel
    tasks = [render_frame.remote(i, job_secret_key) for i in range(N_FRAMES)]
    for msg in ray.get(tasks):
        print(msg)




if __name__ == "__main__":
    main()
