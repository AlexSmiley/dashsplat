#!/usr/bin/env python3
"""Synthesise every sound in the film (no samples, no music library): two ambient beds, a soft drone pad and
the effects. Deterministic (fixed seeds). Writes 48 kHz float WAVs to film/work/audio/.
usage: uv run --with numpy --with scipy python film/sound/synth.py
"""
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'audio')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)


def save(name, x, peak_db=-6.0):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], -1)
    m = np.max(np.abs(x)) + 1e-12
    x = x / m * 10 ** (peak_db / 20)
    # 16-bit with TPDF dither: the long beds stay small on disk
    d = (rng.random(x.shape) - rng.random(x.shape)) / 32768.0
    wavfile.write(os.path.join(OUT, name), SR, np.clip((x + d) * 32767, -32768, 32767).astype(np.int16))
    print(name, f'{len(x) / SR:.2f}s')


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], 'bandpass', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'lowpass', fs=SR, output='sos'), x, axis=0)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'highpass', fs=SR, output='sos'), x, axis=0)


def pink(n, ch=2):
    w = rng.standard_normal((n, ch))
    f = np.fft.rfftfreq(n, 1 / SR)
    s = np.fft.rfft(w, axis=0) / np.maximum(np.sqrt(f), 1.0)[:, None]
    return np.fft.irfft(s, n, axis=0)


def brown(n, ch=2):
    w = rng.standard_normal((n, ch))
    f = np.fft.rfftfreq(n, 1 / SR)
    s = np.fft.rfft(w, axis=0) / np.maximum(f, 5.0)[:, None]
    return np.fft.irfft(s, n, axis=0)


def lfo(n, rate, depth, phase=0.0):
    t = np.arange(n) / SR
    return 1 + depth * np.sin(2 * np.pi * rate * t + phase)


def smooth_noise(n, rate):
    k = max(2, int(n / SR * rate))
    pts = rng.standard_normal(k + 3)
    x = np.linspace(0, k, n)
    return np.interp(x, np.arange(k + 3), pts)


def env(n, a, d, shape=1.0):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / max(d, 1e-4))
    return e ** shape


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


L = 250 * SR  # beds cover the whole film

# car cabin: low road rumble, tyre hiss, a little wind on glass, a faint electric motor whine
n = L
rumble = lp(brown(n), 140, 4) * 1.0
road = bp(pink(n), 180, 900) * (0.55 + 0.25 * smooth_noise(n, 0.7))[:, None]
hiss = hp(pink(n), 2500) * 0.05 * (1 + 0.3 * smooth_noise(n, 0.3))[:, None]
wind = bp(pink(n), 500, 1600) * (0.12 + 0.08 * smooth_noise(n, 0.15))[:, None]
t = np.arange(n) / SR
whine = 0.012 * np.sin(2 * np.pi * (780 + 25 * smooth_noise(n, 0.1)) * t)
cab = norm(rumble) * 0.9 + norm(road) * 0.45 + norm(hiss) * 0.08 + norm(wind) * 0.18 + whine[:, None]
save('amb_cabin.wav', cab, -10)

# machine room: fan noise with two soft resonances, mains hum, a faint coil whine
fan = bp(pink(n), 90, 3500) * 0.6
fan += bp(pink(n), 230, 260, 4) * 2.0 + bp(pink(n), 460, 500, 4) * 1.2
hum = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in [(60, 0.03, 0), (120, 0.05, 1), (180, 0.02, 2), (240, 0.012, 3)])
coil = 0.002 * np.sin(2 * np.pi * 9100 * t) * (1 + 0.5 * smooth_noise(n, 0.5))
room = norm(fan) * 0.6 + hum[:, None] * 1.0 + coil[:, None]
save('amb_room.wav', room, -12)

# drone pad: open fifths on D with a slow-moving ninth, gentle detune and a high airy layer
def voice(f, n, det=0.0015, amp=1.0, pan=0.0):
    t = np.arange(n) / SR
    v = np.zeros(n)
    for k, a in [(1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08)]:
        v += a * (np.sin(2 * np.pi * f * k * (1 + det) * t) + np.sin(2 * np.pi * f * k * (1 - det) * t + 1.3))
    v *= amp * (0.75 + 0.25 * smooth_noise(n, 0.05))
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    return np.stack([v * l, v * r], -1)

pad = voice(73.42, n, amp=0.9, pan=-0.2) + voice(110.0, n, amp=0.55, pan=0.25) + voice(146.83, n, amp=0.35, pan=-0.1)
pad += voice(164.81, n, amp=0.22, pan=0.4) * (0.5 + 0.5 * np.clip(smooth_noise(n, 0.03), -1, 1))[:, None]
air = bp(pink(n), 2400, 5200) * (0.05 + 0.04 * smooth_noise(n, 0.08))[:, None]
pad = lp(pad, 1400, 2) + norm(air) * 0.05 * np.max(np.abs(pad))
save('pad.wav', pad, -8)

# tick: a precise little click
n = int(0.08 * SR)
tt = np.arange(n) / SR
tick = np.sin(2 * np.pi * 2900 * tt) * np.exp(-tt / 0.006) + 0.4 * np.sin(2 * np.pi * 5200 * tt) * np.exp(-tt / 0.003)
tick += hp(rng.standard_normal(n), 3000) * np.exp(-tt / 0.0015) * 0.3
save('tick.wav', tick, -3)

# shutter: mirror slap then curtain close, filtered bursts with a metallic ring
n = int(0.35 * SR)
tt = np.arange(n) / SR
def burst(at, dur, lo, hi, amp):
    x = np.zeros(n)
    i = int(at * SR)
    m = int(dur * SR)
    b = bp(rng.standard_normal(m), lo, hi) * np.exp(-np.arange(m) / SR / (dur / 4))
    x[i:i + m] += b[: n - i] * amp
    return x
sh = burst(0.0, 0.03, 900, 6000, 1.0) + burst(0.055, 0.04, 600, 4500, 0.8)
sh += 0.25 * np.sin(2 * np.pi * 3100 * tt) * np.exp(-tt / 0.01) + 0.2 * np.sin(2 * np.pi * 1700 * np.maximum(tt - 0.055, 0)) * np.exp(-np.maximum(tt - 0.055, 0) / 0.015) * (tt > 0.055)
save('shutter.wav', sh, -3)

# low sweep for wipes: noise through a rising band, swelling and passing left to right
n = int(1.6 * SR)
tt = np.arange(n) / SR
x = pink(n, 1)[:, 0]
out = np.zeros(n)
hop = 1024
for i in range(0, n, hop):
    c = 180 + 900 * (i / n) ** 1.5
    seg = bp(x[max(0, i - 4096): i + hop], c * 0.6, c * 1.6)[-hop:]
    out[i:i + len(seg)] = seg[: n - i]
e = np.sin(np.pi * np.clip(tt / 1.6, 0, 1)) ** 2
pan = np.clip(tt / 1.6, 0, 1)
sw = np.stack([out * e * np.cos(pan * np.pi / 2), out * e * np.sin(pan * np.pi / 2)], -1)
sw += np.stack([lp(out, 200) * e] * 2, -1) * 1.5
save('sweep.wav', sw, -4)

# granular shimmer: tiny sine grains rising in density and pitch over 3.2 s, spread across the stereo field
n = int(3.6 * SR)
sh = np.zeros((n, 2))
for g in range(1400):
    u = rng.random() ** 0.6
    at = u * 3.0
    f = 1800 + 5200 * u * rng.random() + 600 * rng.random()
    d = 0.02 + 0.05 * rng.random()
    m = int(d * SR)
    i = int(at * SR)
    if i + m >= n:
        continue
    gr = np.sin(2 * np.pi * f * np.arange(m) / SR) * np.hanning(m) * (0.3 + 0.7 * rng.random()) * (0.3 + u)
    p = rng.random()
    sh[i:i + m, 0] += gr * np.sqrt(1 - p)
    sh[i:i + m, 1] += gr * np.sqrt(p)
tail = np.exp(-np.maximum(0, np.arange(n) / SR - 3.0) / 0.25)
sh *= tail[:, None]
bed = voice(587.33, n, amp=0.05) * (np.sin(np.pi * np.clip(np.arange(n) / SR / 3.4, 0, 1)) ** 2)[:, None]
save('shimmer.wav', sh + bed, -4)

# sub boom
n = int(4.0 * SR)
tt = np.arange(n) / SR
f = 38 + 26 * np.exp(-tt / 0.25)
ph = 2 * np.pi * np.cumsum(f) / SR
sub = np.sin(ph) * np.exp(-tt / 1.1) * np.minimum(1, tt / 0.01)
sub += lp(rng.standard_normal(n), 120) * np.exp(-tt / 0.3) * 0.3
save('sub.wav', sub, -3)

# power cut: hum and fans drop away under a relay thunk; and the return
n = int(2.2 * SR)
tt = np.arange(n) / SR
drop = np.clip(1 - (tt - 0.15) / 0.35, 0, 1) ** 2
fpitch = 60 * (0.6 + 0.4 * drop)
ph = 2 * np.pi * np.cumsum(fpitch) / SR
humd = (np.sin(ph) * 0.5 + np.sin(2 * ph) * 0.6 + np.sin(3 * ph) * 0.25) * drop
fan = bp(pink(n, 1)[:, 0], 120, 2500) * drop * 0.6
thunk = lp(rng.standard_normal(n), 300) * np.exp(-np.maximum(tt - 0.15, 0) / 0.04) * (tt > 0.15) * 3
save('power_down.wav', humd + norm(fan) * 0.5 + thunk, -4)
n = int(2.6 * SR)
tt = np.arange(n) / SR
rise = np.clip((tt - 0.1) / 1.4, 0, 1)
fpitch = 60 * (0.55 + 0.45 * rise)
ph = 2 * np.pi * np.cumsum(fpitch) / SR
humu = (np.sin(ph) * 0.5 + np.sin(2 * ph) * 0.6 + np.sin(3 * ph) * 0.25) * rise ** 1.5
fan = bp(pink(n, 1)[:, 0], 120, 2500) * rise * 0.6
click = bp(rng.standard_normal(n), 1500, 6000) * np.exp(-tt / 0.004) * 1.5
fade = np.clip((2.6 - tt) / 0.8, 0, 1)
save('power_up.wav', (humu + norm(fan) * 0.5) * fade + click, -4)

# unfold: soft pneumatic hiss with a slide and a quiet click
n = int(1.4 * SR)
tt = np.arange(n) / SR
hiss = hp(rng.standard_normal(n), 2200) * np.exp(-tt / 0.25) * np.minimum(1, tt / 0.02) * 0.5
slide = bp(rng.standard_normal(n), 300, 1200) * np.sin(np.pi * np.clip(tt / 0.9, 0, 1)) ** 2 * 0.5
clk = np.zeros(n); i = int(0.92 * SR); m = int(0.02 * SR)
clk[i:i + m] = bp(rng.standard_normal(m), 1500, 7000) * np.exp(-np.arange(m) / SR / 0.003)
save('unfold.wav', hiss + slide + clk * 0.9, -5)

# car pass-by: doppler-shaped tyre and engine noise crossing left to right
n = int(3.2 * SR)
tt = np.arange(n) / SR
closest = 1.5
dist = np.sqrt(4 + (14 * (tt - closest)) ** 2)
amp = 1 / dist
x = pink(n, 1)[:, 0]
lo_part = bp(x, 120, 700)
hi_part = bp(x, 1500, 5000) * 0.3
sig = (lo_part + hi_part) * amp
pan = 1 / (1 + np.exp(-(tt - closest) * 3))
pb = np.stack([sig * np.cos(pan * np.pi / 2), sig * np.sin(pan * np.pi / 2)], -1)
save('passby.wav', pb, -5)

# data blips: quiet clean chirps
n = int(2.4 * SR)
bl = np.zeros(n)
for k in range(26):
    at = rng.random() * 2.2
    f = rng.choice([1320, 1760, 2093, 2637, 3136])
    m = int(0.035 * SR)
    i = int(at * SR)
    bl[i:i + m] += np.sin(2 * np.pi * f * np.arange(m) / SR) * np.hanning(m) * (0.4 + 0.6 * rng.random())
save('blips.wav', bl, -6)

# dissonant drone for the dead ends: slowly beating detuned partials, hollow
n = int(7.0 * SR)
tt = np.arange(n) / SR
dd = (np.sin(2 * np.pi * 110 * tt) + np.sin(2 * np.pi * 116.5 * tt) * 0.8 + np.sin(2 * np.pi * 155.6 * tt) * 0.4 + np.sin(2 * np.pi * 233 * tt) * 0.15)
dd *= np.sin(np.pi * np.clip(tt / 7.0, 0, 1)) ** 1.5
save('dissonant.wav', lp(dd, 900), -6)
