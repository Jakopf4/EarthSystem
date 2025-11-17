"""Script to create an animation of in-degrees and out-degrees over time."""
import os
import re
import shutil
import subprocess

from inout import plot_degrees

import xarray as xr

# -------------------------------
# User parameters
# -------------------------------
scenario = "245"  # "370", or "585"
fps = 10
output_format = "mp4"  # "gif" or "mp4"
delete_after = True

# -------------------------------
# Paths
# -------------------------------
PLOT_DIR = "../results"
DATA_DIR = "../data"

frame_files = []
years = range(2030, 2099, 1)
months = [1]

for year in years:
    for month in months:
        filename = f"scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
        filepath = os.path.join(DATA_DIR, filename)

        ds = xr.open_dataset(filepath)
        plot_degrees(scenario, year, month, flag="both")
        print(f"  ... saved frame for {year}-{month:02d}")
        frame_files.append(os.path.join(PLOT_DIR, f"frame_{year}_{month:02d}.png"))

frame_list = sorted(
    frame_files,
    key=lambda x: (
        re.search(r"_(\d{2})\.png$", x).group(1),
        re.search(r"_(\d{4})_", x).group(1),
    ),
)

print(f"Found {len(frame_list)} images.")

# FFmpeg needs input pattern like Run_181_Wald_t%04d_Dr.png
# → so we need files named sequentially. Let's symlink them:
tmp_dir = f"./tmp_frames_{scenario}"
os.makedirs(tmp_dir, exist_ok=True)

for i, f in enumerate(frame_list):
    link_name = os.path.join(tmp_dir, f"frame_{i:05d}.png")
    source_file_abs = os.path.abspath(f)

    if not os.path.exists(link_name):
        os.symlink(source_file_abs, link_name)

# Output
if output_format == "gif":
    out_path = f"../results/Scenario_{scenario}_InOutDegrees.gif"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        f"{tmp_dir}/frame_%05d.png",
        out_path,
    ]
else:
    out_path = f"../results/Scenario_{scenario}_InOutDegrees.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        f"{tmp_dir}/frame_%05d.png",
        "-loglevel",
        "error",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]

# Run FFmpeg
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    print(f"✅ Movie saved to {out_path}")
    if delete_after:
        for f in frame_list:
            os.remove(f)
        print(f"🗑️  Deleted {len(frame_list)} PNG files.")
else:
    print("❌ Movie creation failed.")

# Cleanup temporary links
shutil.rmtree(tmp_dir)
