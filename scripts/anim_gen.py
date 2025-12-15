"""Script to create an animation for different plotting functions."""

import os
import re
import shutil
import subprocess

import plot_all as pa


def anim_gen(
    scenario: int, flag: str, fps: int = 30, delete_after: bool = False
) -> None:
    """Create an animation of for different plotting functions.

    Args:
        scenario (int): Scenario identifier.
        flag (str): Type of plot to generate animation for.
        fps (int): Frames per second for the output video.
        delete_after (bool): Whether to delete individual frames after video creation.

    Returns:
        None, but saves an mp4 video file.

    """
    # --- Configuration based on flag ---
    PLOT_DIR = None
    plot_function = None
    plot_name = None
    out_path = None

    if flag == "InOut":
        PLOT_DIR = f"../results/plots/InOut/Scenario{scenario}"
        plot_function = pa.plot_degrees
        plot_name = "degrees"
        out_path = f"../results/Scenario_{scenario}_InOutDegrees.mp4"

    elif flag == "YearInOut":
        PLOT_DIR = f"../results/plots/YearInOut/Scenario{scenario}"
        plot_function = pa.plot_yearly_degrees
        plot_name = "yearly_degrees"
        out_path = f"../results/Scenario_{scenario}_YearInOutDegrees.mp4"

    elif flag == "DiffYearInOut":
        PLOT_DIR = f"../results/plots/YearInOut/Scenario{scenario}"
        plot_function = pa.plot_diff_yearly_degrees
        plot_name = "yearly_diff_degrees"
        out_path = f"../results/Scenario_{scenario}_YearDiffInOutDegrees.mp4"

    elif flag == "Clustering":
        PLOT_DIR = f"../results/plots/Clustering/Scenario{scenario}"
        plot_function = pa.plot_clustering
        plot_name = "clustering"
        out_path = f"../results/Scenario_{scenario}_Clustering.mp4"

    elif flag == "FFL":
        PLOT_DIR = f"../results/plots/FFL/Scenario{scenario}"
        plot_function = pa.plot_ffl
        plot_name = "ffl"
        out_path = f"../results/Scenario_{scenario}_FFL.mp4"

    elif flag == "Deforestation":
        PLOT_DIR = f"../results/plots/Deforestation"
        plot_function = pa.plot_deforestation
        plot_name = "deforestation"
        out_path = f"../results/Deforestation.mp4"

    if PLOT_DIR is None:
        exit(1)

    frame_files = []
    years = range(2030, 2100, 1)
    if flag == "Deforestation":
        years = range(2002, 2051, 1)
    months = range(1, 13, 1)

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # --- Logic for monthly plots ---
    if (
        plot_function != pa.plot_yearly_degrees
        and plot_function != pa.plot_diff_yearly_degrees
        and plot_function != pa.plot_deforestation
    ):
        for year in years:
            for month in months:
                file_path = os.path.join(
                    PLOT_DIR, f"{plot_name}_{scenario}_{year}_{month:02d}.png"
                )

                # Check if plot already exists
                if os.path.exists(file_path):
                    print(
                        f"  ... plot for {year}-{month:02d} already exists. Skipping."
                    )
                    frame_files.append(file_path)
                    continue
                else:
                    try:
                        plot_function(scenario, year, month)
                        print(f"  ... saved frame for {year}-{month:02d}")
                        frame_files.append(file_path)
                    except Exception as e:
                        print(
                            f"❌ ERROR: Plotting failed for {year}-{month:02d}. Error: {e}"
                        )
                        continue

    # --- Logic for yearly deforestation plots ---
    elif plot_function == pa.plot_deforestation:
        for year in years:
            file_path = os.path.join(PLOT_DIR, f"{plot_name}_{year}.png")

            if os.path.exists(file_path):
                print(f"  ... plot for {year} already exists. Skipping.")
                frame_files.append(file_path)
                continue
            else:
                try:
                    plot_function(year)
                    print(f"  ... saved frame for {year}")
                    frame_files.append(file_path)
                except Exception as e:
                    print(f"❌ ERROR: Plotting failed for {year}. Error: {e}")
                    continue

    # --- Logic for yearly plots ---
    else:
        for year in years:
            file_path = os.path.join(PLOT_DIR, f"{plot_name}_{scenario}_{year}.png")

            if os.path.exists(file_path):
                print(f"  ... plot for {year} already exists. Skipping.")
                frame_files.append(file_path)
                continue
            else:
                try:
                    plot_function(scenario, year)
                    print(f"  ... saved frame for {year}")
                    frame_files.append(file_path)
                except Exception as e:
                    print(f"❌ ERROR: Plotting failed for {year}. Error: {e}")
                    continue

    print(f"\nCollected {len(frame_files)} frame files.")


    # --- Sort by Years only ---
    if (
        plot_function == pa.plot_yearly_degrees
        or plot_function == pa.plot_diff_yearly_degrees
        or plot_function == pa.plot_deforestation
    ):
        frame_list = sorted(
            frame_files,
            key=lambda x: re.search(r"_(\d{4})\.png$", x).group(1),
        )

    # --- Sort by Years and Months ---
    else:
        frame_list = sorted(
            frame_files,
            key=lambda x: (
                re.search(r"_(\d{4})_", x).group(1),
                re.search(r"_(\d{2})\.png$", x).group(1),
            ),
        )

    print(f"Found {len(frame_list)} images.")

    # --- Temporary Directory Cleanup and Symlink Creation ---
    tmp_dir = f"./tmp_frames_{scenario}"

    if os.path.exists(tmp_dir):
        print(f"🗑️  Cleaning up stale temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    for i, f in enumerate(frame_list):
        link_name = os.path.join(tmp_dir, f"frame_{i:05d}.png")

        source_file_abs = os.path.abspath(f)

        if not os.path.exists(link_name):
            os.symlink(source_file_abs, link_name)

    # --- Create the Movie using ffmpeg ---
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
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

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # --- Finalization and Cleanup ---
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ Movie saved to {out_path}")
        if delete_after:
            for f in frame_list:
                os.remove(f)
            print(f"🗑️  Deleted {len(frame_list)} PNG files.")
    else:
        print("❌ Movie creation failed.")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    scenario = 585
    # anim_gen(scenario, "DiffYearInOut", fps=10, delete_after=False)
    anim_gen(scenario, "Deforestation", delete_after=False)
    # anim_gen(scenario, "FFL")
