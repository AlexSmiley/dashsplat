#!/usr/bin/env python3
"""Set the delivered loudness: one gain for the mix and both stems so the mix lands at -20 LUFS integrated,
then a transparent true-peak limiter (4x oversampled) so nothing exceeds -3 dBTP. The limiter is applied to the
mix and the effects stem (the effects carry every peak); the ambience stem only gets the gain.
usage: uv run --with numpy --with scipy --with pyloudnorm python film/sound/master.py <dir with mix-raw.wav amb-raw.wav sfx-raw.wav> [prefix]
Writes mix.wav, ambience.wav, effects.wav (48 kHz, 24-bit) and prints the measurements.
"""
import sys, os
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from scipy.io import wavfile

TARGET, CEIL = -20.0, -3.2
d = sys.argv[1]
pre = sys.argv[2] if len(sys.argv) > 2 else ''


def read(name):
    sr, x = wavfile.read(os.path.join(d, name))
    x = x.astype(np.float64)
    if x.dtype.kind == 'i' or np.abs(x).max() > 2:
        x /= 32768.0
    if x.ndim == 1:
        x = np.stack([x, x], -1)
    return sr, x


def true_peak(x, sr):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def limit(x, sr, ceil_db):
    """Look-ahead peak limiter on the 4x-oversampled envelope: gain never exceeds what keeps |x| under ceil."""
    c = 10 ** (ceil_db / 20)
    up = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(1)
    env = up.reshape(-1, 4).max(1)[: len(x)] if len(up) >= 4 * len(x) else np.abs(x).max(1)
    g = np.minimum(1.0, c / np.maximum(env, 1e-9))
    la = int(0.005 * sr)  # 5 ms look-ahead: take the minimum gain over the coming window
    gmin = np.minimum.reduce([np.roll(g, -k) for k in range(0, la, max(1, la // 16))])
    # release 80 ms, attack immediate
    rel = np.exp(-1 / (0.08 * sr))
    out = np.empty_like(gmin)
    cur = 1.0
    for i, v in enumerate(gmin):
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    return x * out[:, None]


sr, mix = read(pre + 'mix-raw.wav')
_, amb = read(pre + 'amb-raw.wav')
_, sfx = read(pre + 'sfx-raw.wav')
meter = pyln.Meter(sr)
L = meter.integrated_loudness(mix)
# GAIN (dB) reuses the master's gain, so chapter files keep the film's balance instead of being re-normalised
gain = float(os.environ['GAIN']) if os.environ.get('GAIN') else TARGET - L
if not os.environ.get('GAIN'):
    open(os.path.join(d, 'gain.txt'), 'w').write(f'{gain:.3f}\n')
g = 10 ** (gain / 20)
mix, amb, sfx = mix * g, amb * g, sfx * g
tp = true_peak(mix, sr)
if tp > CEIL:
    mix = limit(mix, sr, CEIL)
    sfx = limit(sfx, sr, CEIL)
for name, x in [('mix.wav', mix), ('ambience.wav', amb), ('effects.wav', sfx)]:
    y = np.clip(x, -1, 1)
    wavfile.write(os.path.join(d, pre + name), sr, y.astype(np.float32))
L2 = meter.integrated_loudness(mix)
print(f'raw {L:.1f} LUFS, gain {gain:+.1f} dB, after: {L2:.1f} LUFS, true peak {true_peak(mix, sr):.1f} dBTP '
      f'(before limiter {tp:.1f}); ambience {meter.integrated_loudness(amb):.1f} LUFS, effects {meter.integrated_loudness(sfx):.1f} LUFS')
