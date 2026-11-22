"""Summarise decoded proof length and reprojection-confirmed luminance change at both kinds of handover."""

from pathlib import Path
import json, subprocess, statistics


def read(p):
    return json.loads(p.read_text())


def one(folder, name, start, end):
    path = folder / f"{name}-{start}-{end}.mp4"
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    fs = read(folder / (name + ".json"))["frames"]
    assert (
        int(info["nb_frames"]),
        float(info["duration"]),
        info["width"],
        info["height"],
        info["r_frame_rate"],
    ) == ((end - start) * 30, end - start, 1440, 900, "30/1")
    assert len(fs) == (end - start) * 30
    assert all(abs(f["t"] - (start + i / 30)) < 1e-9 for i, f in enumerate(fs))
    answers = {}
    for key in ["dominantSwitch", "pairBoundary"]:
        result = {}
        for flag in [False, True]:
            frames = [f for f in fs if f["err"] and f[key] == flag]
            ls = []
            for i in range(3):
                obs = [f["err"][i] for f in frames if f["err"][i][0] > 0]
                tot = [sum(a[j] for a in obs) for j in range(3)]
                ls.append(
                    dict(
                        frames=len(obs),
                        samples=tot[0],
                        mean=tot[1] / tot[0],
                        meanPerFrame=statistics.mean(a[1] / a[0] for a in obs),
                        fracOver20=tot[2] / tot[0],
                    )
                )
            result["transition" if flag else "between"] = ls
        answers[key] = result
    f = max(
        (f for f in fs if f["dominantSwitch"] and f["err"][0][0] > 200),
        key=lambda f: f["err"][0][1] / f["err"][0][0],
    )
    answers["worstDominantSwitch"] = dict(
        t=f["t"], mean=f["err"][0][1] / f["err"][0][0]
    )
    answers["video"] = info
    answers["meanStaticFraction"] = statistics.mean(
        f["static_pixels"] / (1440 * 900) for f in fs
    )
    return answers


if __name__ == "__main__":
    folder = Path(__file__).parent / "report-chase"
    a = {
        name: one(folder, name, s, e)
        for name, s, e in [("road", 105, 120), ("exit", 93, 104)]
    }
    (folder / "motion.json").write_text(json.dumps(a, indent=2) + "\n")
    print(json.dumps(a, indent=2))
