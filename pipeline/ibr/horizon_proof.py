"""Hybrid reference player. Inputs: donor manifest, host PLY scene, actual host chase-track, native splat rasterizer."""

from . import layer_player as lp
from .layer_player import Player, norm, readj, sm
import cv2 as cv, numpy as np, sys, subprocess, json, hashlib
from pathlib import Path
from functools import lru_cache
from scipy.spatial.transform import Rotation

W, H, F = lp.W, lp.H, lp.F
SITES = [100.0, 112.0, 118.7, 128.0, 135.0]
cv.setNumThreads(1)


def state(d, t):
    return np.array([np.interp(t, d["t"], d[k]) for k in ["x", "y", "z", "yaw"]])


def sample(im, r, el):
    a = np.unwrap(np.unwrap(np.arctan2(r[:, :, 1], r[:, :, 0]), axis=1), axis=0)
    ee = np.rad2deg(np.arctan2(r[:, :, 2], np.linalg.norm(r[:, :, :2], axis=2)))
    h, w = im.shape[:2]
    uv = np.stack(
        [
            (0.5 + a / (2 * np.pi)) * w - 0.5,
            np.clip((el[1] - ee) / (el[1] - el[0]) * h - 0.5, 0, h - 1),
        ],
        2,
    ).astype("f")
    return cv.remap(
        im, cv.resize(uv, (W, H)), None, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP
    ), ee


class Hybrid(Player):
    def __init__(self, root, host, track, lib):
        super().__init__(root, lib)
        self.host = Path(host)
        self.hd = readj(self.host / "drive.json")
        self.dd = readj(self.asset(self.doc["drive"]))
        self.hd["yaw"] = np.unwrap(self.hd["yaw"])
        self.dd["yaw"] = np.unwrap(self.dd["yaw"])
        self.ht = {}
        for p in readj(Path(track))["frames"]:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(p["quat"]).as_matrix() @ np.diag([1, -1, -1])
            T[:3, 3] = p["pos"]
            self.ht[round(p["t"] * 30)] = T
        self.tiles = readj(self.host / "map.json")["tiles"]
        self.bb = np.array([e["bbox"] for e in self.tiles])
        assert {"ground", "objects"} <= {e.get("layer") for e in self.tiles}
        self.fade = self.doc.get("backdropRange", {}).get("splatFadeM", [40, 60])
        self.mid = None

    @lru_cache(64)
    def tile(self, f):
        e = self.tiles[f]
        with (self.host / e["file"]).open("rb") as stream:
            while True:
                line = stream.readline()
                assert line
                if line.strip() == b"end_header":
                    break
            a = np.fromfile(stream, "<f4").reshape(-1, 24)
            assert len(a) == e["count"]
            return np.column_stack(
                [a, np.full(len(a), e.get("layer") == "objects", "f")]
            )

    def view(self, t):
        host = self.ht[round(t * 30)]
        ds = state(self.dd, t)
        hs = state(self.hd, t)
        R = Rotation.from_euler("z", ds[3] - hs[3]).as_matrix()
        T = host.copy()
        T[:3, :3] = R @ host[:3, :3]
        T[:3, 3] = ds[:3] + R @ (host[:3, 3] - hs[:3])
        return host, T

    def background(self, t, T, weights, mid=None):
        wr = lp.RAYS[1::3, 1::3] @ T[:3, :3].astype("f").T
        O = T[:3, 3]
        one = self.pan[next(iter(weights))]
        sky, _ = sample(self.rgb(one["sky"]), wr, self.doc["skyElevation"])
        if any(self.pan[k]["sky"] != one["sky"] for k in weights):
            sky = sum(
                w
                * sample(self.rgb(self.pan[k]["sky"]), wr, self.doc["skyElevation"])[0]
                for k, w in weights.items()
            )
        bg = sky.astype(
            "f"
        )  # An untextured underlay only through holes below the horizon; ground PLY colours remain measured.
        elev = cv.resize(
            np.arctan2(wr[:, :, 2], np.linalg.norm(wr[:, :, :2], axis=2)).astype("f"),
            (W, H),
        )
        base = sm((-elev - 0.018) / 0.085)[:, :, None]
        bg += (np.array([99, 106, 102], "f") - bg) * base
        zm = np.full((H, W), 9999, "f")
        profiles = []
        afar = None
        for typ in [False, True] if mid else [False]:
            rgb = np.zeros_like(bg)
            a = np.zeros((H, W), "f")
            zs = np.zeros_like(a)
            for k, w in weights.items():
                p = self.pan[k]
                r = p["radius"]
                im = self.rgba(p["rgb"], p["matte"])
                origin = p["origin"]
                if typ:
                    pair = mid.get(k)
                    if pair is None:
                        continue
                    r = 40.0
                    stem, origin = pair
                    im = self.testtex(stem)
                v = O - np.array(origin)
                q = np.sum(wr[:, :, :2] ** 2, 2)
                b = wr[:, :, :2] @ v[:2]
                disc = b * b - q * (v[:2] @ v[:2] - r * r)
                z = (-b + np.sqrt(np.maximum(0, disc))) / np.maximum(0.001, q)
                rel = v + wr * z[:, :, None]
                c, el = sample(im, rel, self.doc["backdropElevation"])
                lo, hi = self.doc["backdropElevation"]
                ok = cv.resize(
                    ((disc > 0) & (z > 0.3) & (el >= lo) & (el <= hi)).astype("f"),
                    (W, H),
                )
                al = c[:, :, 3] * ok / 255.0
                rgb += w * ok[:, :, None] * c[:, :, :3]
                a += w * al
                zs += w * al * cv.resize(z.astype("f"), (W, H))
                if not typ and w > 0.08:
                    profiles.append(
                        (w, np.argmax(al > 0.60, axis=0), np.max(al, axis=0) > 0.6)
                    )
            z = zs / np.maximum(0.001, a)
            on = (a > 0.0001) & (z < zm + 0.15)
            bg = bg * (1 - (a * on)[:, :, None]) + rgb * on[:, :, None]
            zm = np.where(on & (a > 0.50), z, zm)
            afar = a if afar is None else afar
        return bg, zm, afar, profiles

    @lru_cache(16)
    def testtex(self, stem):
        a = cv.imread(str(stem) + ".png", 0)
        rgb = cv.imread(str(stem) + ".jpg")[:, :, ::-1]
        return np.dstack([np.uint8(rgb * (a[:, :, None] / 255) + 0.5), a])

    def splats(self, t, T, fade):
        eye = T[:3, 3]
        d = np.linalg.norm(
            np.maximum(
                0, np.maximum(self.bb[:, :2] - eye[:2], eye[:2] - self.bb[:, 3:5])
            ),
            axis=1,
        )
        parts = []
        for i in np.flatnonzero(d < fade[1] + 3):
            a = self.tile(int(i))
            delta = a[:, :3] - eye
            dist = np.hypot(delta[:, 0], delta[:, 1])
            keep = (
                (dist < fade[1])
                & (delta @ T[:3, 2] > -0.3)
                & (a[:, 17] <= t)
                & (a[:, 18] > t)
            )
            p = a[keep].copy()
            v = (1 - sm((dist[keep] - fade[0]) / (fade[1] - fade[0]))) / (
                1 + np.exp(-p[:, 9])
            )
            p[:, 9] = np.log(np.maximum(0.00001, v) / np.maximum(0.00001, 1 - v))
            parts.append(p[v > 0.002])
        pts = (
            np.ascontiguousarray(np.concatenate(parts), "f")
            if parts
            else np.empty((0, 25), "f")
        )
        buf = np.zeros((H, W, 6), "f")
        rw = np.ascontiguousarray(np.repeat(T[None], 33, axis=0), "d")
        intr = np.array([F, F, W / 2, H / 2, 0, 0, 0, 0], "d")
        self.lib.raster(
            pts.ctypes.data,
            len(pts),
            t,
            3,
            rw.ctypes.data,
            intr.ctypes.data,
            0,
            W,
            H,
            buf.ctypes.data,
        )
        return buf

    def draw(self, t, diagnose=False, fade=None):
        host, T = self.view(t)
        weights, _ = self.selection(t, T)
        bg, z, a, profiles = self.background(t, T, weights, self.mid)
        buf = self.splats(t, host, fade or self.fade)
        alpha = buf[:, :, 3]
        sd = buf[:, :, 4] / np.maximum(0.0001, alpha * np.linalg.norm(lp.RAYS, axis=2))
        ok = sd < z + 0.15
        im = bg * (1 - (alpha * ok)[:, :, None]) + buf[:, :, :3] * ok[:, :, None]
        d = dict(
            T=host,
            weighted=weights,
            depth=z,
            alpha=a,
            foreground=alpha,
            object=buf[:, :, 5],
            profiles=profiles,
        )
        return np.uint8(np.clip(im + 0.5, 0, 255)), d


def save(p, img):
    assert cv.imwrite(str(p), img[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 93])


def stills(h, out):
    pics = []
    sites = []
    for t in SITES:
        im, d = h.draw(t)
        save(out / f"{t:.1f}.jpg", im)
        p = cv.resize(im, (600, 375), interpolation=cv.INTER_AREA)
        cv.putText(p, f"{t:.1f} s", (12, 28), 0, 0.7, (250, 250, 250), 2, cv.LINE_AA)
        pics.append(p)
        sites.append(
            dict(
                t=t,
                panos=d["weighted"],
                nearObjectPixels=int((d["object"] > 0.35).sum()),
                skylineMedianRow=int(np.median(np.argmax(d["alpha"] > 0.65, axis=0))),
            )
        )
    pics.append(np.zeros_like(pics[0]))
    save(
        out / "hybrid-witnesses.jpg",
        np.vstack([np.hstack(pics[i : i + 2]) for i in range(0, 6, 2)]),
    )
    (out / "sites.json").write_text(json.dumps(sites))
    print(sites, flush=True)


def motion(prev, im, d):
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    ok = (d["alpha"] > 0.8) & (d["foreground"] < 0.18) & (d["depth"] < 300)
    y = np.arange(H)[:, None]
    h = np.argmax(d["alpha"] > 0.55, axis=0)
    roi = ok & (y > h - 3) & (y < h + 70)
    pts = cv.goodFeaturesToTrack(gray, 180, 0.01, 7, mask=roi.astype("B"))
    cur = (
        gray,
        d["T"],
        pts,
        d["depth"][np.rint(pts[:, 0, 1]).astype(int), np.rint(pts[:, 0, 0]).astype(int)]
        if pts is not None
        else [],
        d["weighted"],
    )
    res = []
    if prev is not None and prev[2] is not None:
        old, P, p, z, ws = prev
        q, st, _ = cv.calcOpticalFlowPyrLK(
            old, gray, p, None, winSize=(17, 17), maxLevel=2
        )
        back, st2, _ = cv.calcOpticalFlowPyrLK(
            gray, old, q, None, winSize=(17, 17), maxLevel=2
        )
        xy = p[:, 0]
        wp = (
            np.c_[
                (xy[:, 0] + 0.5 - W / 2) / F,
                (xy[:, 1] + 0.5 - H / 2) / F,
                np.ones(len(p)),
            ]
            * z[:, None]
        ) @ P[:3, :3].T + P[:3, 3]
        v = (wp - d["T"][:3, 3]) @ d["T"][:3, :3]
        want = v[:, :2] / v[:, 2, None] * F + [W / 2 - 0.5, H / 2 - 0.5]
        ix = np.rint(q[:, 0]).astype(int)
        good = (
            (st[:, 0] * st2[:, 0] > 0)
            & (np.linalg.norm(back[:, 0] - xy, axis=1) < 1.2)
            & (v[:, 2] > 0.3)
            & (ix[:, 0] >= 0)
            & (ix[:, 0] < W)
            & (ix[:, 1] >= 0)
            & (ix[:, 1] < H)
        )
        good &= ok[ix[:, 1].clip(0, H - 1), ix[:, 0].clip(0, W - 1)]
        res = np.linalg.norm(q[:, 0] - want, axis=1)[good].tolist()
    return cur, res


def film(h, out):
    file = out / "hybrid-105-120.mp4"
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1440x900",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "16M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(file),
        ],
        stdin=subprocess.PIPE,
    )
    prev = None
    err = [[], []]
    sep = []
    sheets = []
    for i in range(450):
        t = 105 + i / 30
        im, d = h.draw(t)
        enc.stdin.write(im.tobytes())
        switch = prev is not None and max(prev[-1], key=prev[-1].get) != max(
            d["weighted"], key=d["weighted"].get
        )
        prev, r = motion(prev, im, d)
        err[int(switch)].extend(r)
        pr = sorted(d["profiles"], key=lambda q: -q[0])
        if len(pr) > 1:
            _, y1, on = pr[0]
            _, y2, on2 = pr[1]
            xx = np.arange(W)
            keep = (
                on
                & on2
                & (d["foreground"][y1, xx] < 0.2)
                & (d["foreground"][y2, xx] < 0.2)
            )
            sep.extend(abs(y1[keep][::9] - y2[keep][::9]).tolist())
        if i % 30 == 0:
            sheets.append(cv.resize(im, (384, 240), interpolation=cv.INTER_AREA))
        if i % 120 == 0:
            print("film", i, flush=True)
    enc.stdin.close()
    assert enc.wait() == 0
    vid = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height,duration,r_frame_rate",
                "-of",
                "json",
                str(file),
            ]
        )
    )["streams"][0]
    assert (
        vid["nb_frames"],
        vid["width"],
        vid["height"],
        vid["duration"],
        vid["r_frame_rate"],
    ) == ("450", 1440, 900, "15.000000", "30/1")
    save(
        out / "motion-sheet.jpg",
        np.vstack([np.hstack(sheets[i : i + 3]) for i in range(0, 15, 3)]),
    )
    summ = lambda v: (
        dict(n=len(v), p50=float(np.median(v)), p90=float(np.percentile(v, 90)))
        if v
        else None
    )
    stats = dict(
        sha256=hashlib.sha256(h.asset("ibr.json").read_bytes()).hexdigest(),
        video=vid,
        slipPxPerFrame=summ(err[0]),
        switchSlipPxPerFrame=summ(err[1]),
        skylineSeparationPx=summ(sep),
    )
    (out / "film.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    root, host, track, library, path, cmd = sys.argv[1:]
    h = Hybrid(root, host, track, library)
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    (stills if cmd == "stills" else film)(h, out)
