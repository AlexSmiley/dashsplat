#!/usr/bin/env python3
"""Ingest Tesla dashcam clips.

Reads data/clips/<clip>/<clip>-<cam>.mp4 and writes data/drive/:
  frames/<cam>/<n:06d>.jpg   every decoded frame, n = running index per camera
  cams/<cam>.json            per-frame table: n, clip, idx, pts, seq, t, SEI fields, ego pose
  timeline.npz               master clock (front camera): t, seq, x, y, yaw, speed, ...
  meta.json                  fps, origin lat/lon, per-camera seq offsets, image sizes

Each packet (access unit) carries its own SEI, so metadata is paired with the
frame it was muxed with, not by counting NALs (some files hold SEIs with no
slice, and ffmpeg's image2 muxer drops frames with duplicate dts).

Conventions: world = ENU metres (x east, y north, z up), origin at the first
front frame. yaw = counter-clockwise from east, radians. Tesla heading_deg is
clockwise from north.
"""
import av, cv2, json, math, os, struct, sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor

CAMS = ['front', 'back', 'left_repeater', 'right_repeater', 'left_pillar', 'right_pillar']
NAMES = {1: 'ver', 2: 'gear', 3: 'seq', 4: 'speed', 5: 'accel_pedal', 6: 'steer', 7: 'blink_l',
         8: 'blink_r', 9: 'brake', 10: 'ap', 11: 'lat', 12: 'lon', 13: 'heading',
         14: 'ax', 15: 'ay', 16: 'az'}


def unescape(b):
    out = bytearray(); z = 0
    for x in b:
        if z >= 2 and x == 3:
            z = 0; continue
        out.append(x); z = z + 1 if x == 0 else 0
    return bytes(out)


def varint(b, i):
    v = 0; s = 0
    while True:
        x = b[i]; i += 1; v |= (x & 0x7f) << s; s += 7
        if not x & 0x80:
            return v, i


def parse_proto(b):
    f = {}; i = 0
    try:
        while i < len(b):
            k, i = varint(b, i); fn = k >> 3; wt = k & 7
            if wt == 0: v, i = varint(b, i)
            elif wt == 5: v = struct.unpack('<f', b[i:i + 4])[0]; i += 4
            elif wt == 1: v = struct.unpack('<d', b[i:i + 8])[0]; i += 8
            elif wt == 2:
                l, i = varint(b, i); i += l; continue
            else: break
            f[NAMES.get(fn, str(fn))] = v
    except (IndexError, struct.error):
        pass
    return f


def parse_packet(buf):
    """Return (sei dict or None, has_slice) for one AVCC access unit."""
    p = 0; sei = None; has_slice = False; n = len(buf)
    while p + 4 <= n:
        ln = struct.unpack('>I', buf[p:p + 4])[0]
        if ln == 0 or p + 4 + ln > n: break
        nal = buf[p + 4:p + 4 + ln]; t = nal[0] & 0x1f
        if t in (1, 5): has_slice = True
        elif t == 6 and sei is None:
            b = unescape(nal); k = 3
            while k < len(b) and b[k] == 0x42: k += 1
            if k > 3 and k < len(b) and b[k] == 0x69:
                sei = parse_proto(b[k + 1:-1])
        p += 4 + ln
    return sei, has_slice


def do_file(args):
    path, outdir, start_n, quality = args
    os.makedirs(outdir, exist_ok=True)
    c = av.open(path); st = c.streams.video[0]
    st.thread_type = 'AUTO'
    rows = []; pending = collections.defaultdict(collections.deque); n = start_n; idx = 0
    def emit(frame):
        nonlocal n, idx
        # pair by pts: a few packets never produce a frame, so order alone would shift the metadata
        sei, pts = pending[frame.pts].popleft()
        img = frame.to_ndarray(format='bgr24')
        cv2.imwrite(os.path.join(outdir, f'{n:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        r = {'n': n, 'idx': idx, 'pts': pts}; r.update(sei or {})
        rows.append(r); n += 1; idx += 1
    for pkt in c.demux(st):
        if pkt.size:
            sei, has_slice = parse_packet(bytes(pkt))
            if not has_slice: continue
            pending[pkt.pts].append((sei, float(pkt.pts * st.time_base)))
        for fr in pkt.decode():
            emit(fr)
    lost = sum(len(q) for q in pending.values())
    return path, rows, (st.width, st.height), lost


def main(root):
    clips = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'clips', '*')) if os.path.isdir(p))
    out = os.path.join(root, 'drive'); os.makedirs(out, exist_ok=True)
    counts_path = os.path.join(out, 'raw_tables.json')
    if os.path.exists(counts_path) and '--force' not in sys.argv:
        raw = json.load(open(counts_path))
    else:
        # pass 1: count frames per file so running indices are known up front
        jobs = []
        for cam in CAMS:
            n = 0
            for clip in clips:
                path = os.path.join(root, 'clips', clip, f'{clip}-{cam}.mp4')
                c = av.open(path); st = c.streams.video[0]
                cnt = sum(1 for pkt in c.demux(st) if pkt.size and parse_packet(bytes(pkt))[1])
                # running indices leave a gap where a packet yields no frame
                jobs.append((path, os.path.join(out, 'frames', cam), n, 95)); n += cnt
        raw = {cam: [] for cam in CAMS}; sizes = {}
        with ProcessPoolExecutor(12) as ex:
            for path, rows, size, lost in ex.map(do_file, jobs):
                clip = os.path.basename(os.path.dirname(path)); cam = os.path.basename(path)[len(clip) + 1:-4]
                for r in rows: r['clip'] = clip
                raw[cam] += rows; sizes[cam] = size
                print('decoded', os.path.basename(path), len(rows), 'undecodable packets', lost, flush=True)
        for cam in CAMS: raw[cam].sort(key=lambda r: r['n'])
        raw['_sizes'] = sizes
        json.dump(raw, open(counts_path, 'w'))
    sizes = raw['_sizes']

    # ---- master clock from the front camera
    F = [r for r in raw['front'] if 'seq' in r and 'lat' in r]
    seq = np.array([r['seq'] for r in F], dtype=np.int64)
    # fps from pts inside clips
    num = den = 0.0
    for clip in clips:
        rs = [r for r in F if r['clip'] == clip]
        num += rs[-1]['seq'] - rs[0]['seq']; den += rs[-1]['pts'] - rs[0]['pts']
    fps = num / den
    seq0 = int(seq[0]); t = (seq - seq0) / fps
    g = lambda k, d=0.0: np.array([r.get(k, d) for r in F], dtype=np.float64)
    speed = g('speed'); heading = np.unwrap(np.radians(g('heading')))
    lat = g('lat'); lon = g('lon')
    R = 6378137.0; lat0 = lat[0]; lon0 = lon[0]
    gx = np.radians(lon - lon0) * R * math.cos(math.radians(lat0)); gy = np.radians(lat - lat0) * R
    dt = np.diff(t, prepend=t[0])
    vm = np.concatenate([[speed[0]], 0.5 * (speed[1:] + speed[:-1])])
    hm = np.concatenate([[heading[0]], 0.5 * (heading[1:] + heading[:-1])])
    x = np.cumsum(vm * dt * np.sin(hm)); y = np.cumsum(vm * dt * np.cos(hm))
    yaw = math.pi / 2 - heading
    np.savez(os.path.join(out, 'timeline.npz'), t=t, seq=seq, x=x, y=y, yaw=yaw, speed=speed,
             heading=heading, gps_x=gx, gps_y=gy, lat=lat, lon=lon, steer=g('steer'),
             ax=g('ax'), ay=g('ay'), az=g('az'), ap=g('ap'), brake=g('brake'),
             blink_l=g('blink_l'), blink_r=g('blink_r'), accel_pedal=g('accel_pedal'), gear=g('gear'))

    # ---- per-camera seq offset relative to front, from identical telemetry tuples
    key = lambda r: (r.get('lat'), r.get('lon'), r.get('speed'), r.get('heading'), r.get('ax'))
    fk = collections.defaultdict(list)
    for r in F:
        if r.get('speed', 0) > 1.0: fk[key(r)].append(r['seq'])
    offsets = {'front': 0.0}
    for cam in CAMS[1:]:
        cnt = collections.Counter()
        for r in raw[cam]:
            for s in fk.get(key(r), []): cnt[r['seq'] - s] += 1
        mode = cnt.most_common(1)[0][0]
        near = {k: v for k, v in cnt.items() if abs(k - mode) <= 2}
        offsets[cam] = sum(k * v for k, v in near.items()) / sum(near.values())
        print(cam, 'seq offset', round(offsets[cam], 3), dict(sorted(near.items())))

    os.makedirs(os.path.join(out, 'cams'), exist_ok=True)
    for cam in CAMS:
        rows = []
        for r in raw[cam]:
            if 'seq' not in r: continue
            tc = (r['seq'] - offsets[cam] - seq0) / fps
            row = dict(r); row['t'] = tc
            row['ego_x'] = float(np.interp(tc, t, x)); row['ego_y'] = float(np.interp(tc, t, y))
            row['ego_yaw'] = float(np.interp(tc, t, yaw)); row['ego_speed'] = float(np.interp(tc, t, speed))
            rows.append(row)
        json.dump(rows, open(os.path.join(out, 'cams', cam + '.json'), 'w'))
        print(cam, 'frames', len(raw[cam]), 'with sei', len(rows), 't', round(rows[0]['t'], 3), round(rows[-1]['t'], 3))
    json.dump({'fps': fps, 'seq0': seq0, 'origin_lat': lat0, 'origin_lon': lon0, 'seq_offsets': offsets,
               'sizes': sizes, 'cams': CAMS, 'clips': clips, 'duration_s': float(t[-1]),
               'distance_m': float(np.sum(vm * dt))}, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('fps', fps, 'duration', t[-1], 'distance', np.sum(vm * dt), 'DR end', x[-1], y[-1], 'GPS end', gx[-1], gy[-1])


if __name__ == '__main__':
    main(sys.argv[1])
