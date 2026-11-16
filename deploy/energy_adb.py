#!/usr/bin/env python3
"""Energy per inference on Android, read through the debugging shell (adb), from Termux on the same phone.

Apps cannot read the battery current sensor, but the adb shell can. With Wireless debugging the phone is not
connected to a cable, so it runs on its battery while we measure.

    python energy_adb.py --check                 # 5 s: can the sensor be read?
    python energy_adb.py                         # ~10 min: 3 x (60 s idle, 120 s model)

Prepare: Wi-Fi on (needed for wireless debugging), mobile data off, screen at a fixed low brightness,
other apps closed, phone unplugged, battery above 30%. Do not touch the phone during the run."""
import argparse, json, os, re, subprocess, sys, threading, time
import numpy as np

BAT = '/sys/class/power_supply/battery'

def adb(*args, serial=None, timeout=15):
    cmd = ['adb'] + (['-s', serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def to_amps(i):                       # kernels report microamps (most) or milliamps
    i = abs(float(i)); return i / 1e6 if i > 20000 else i / 1e3

def to_volts(v):                      # microvolts (most) or millivolts
    v = float(v); return v / 1e6 if v > 100000 else v / 1e3 if v > 1000 else v

CUR_CANDIDATES = ['battery/current_now', 'mtk-gauge/current_now', 'sc27xx-fgu/current_now', 'bms/current_now', 'battery/current_avg',
                  'mtk-gauge/current_avg', 'bms/current_avg']
VOLT_CANDIDATES = ['battery/voltage_now', 'mtk-gauge/voltage_now', 'sc27xx-fgu/voltage_now', 'bms/voltage_now', 'battery/batt_vol']

def discover(serial):
    """Find the first readable current file and voltage source; MediaTek phones often allow only mtk-gauge."""
    def readable(rel):
        r = adb('shell', f'cat /sys/class/power_supply/{rel}', serial=serial)
        v = r.stdout.strip()
        return v if r.returncode == 0 and v.lstrip('-').isdigit() else None
    # Fallback for any vendor (Unisoc sc27xx-fgu, Samsung, ...): every supply that exposes current/voltage files
    ls = adb('shell', 'ls /sys/class/power_supply/', serial=serial).stdout.split()
    extra_c = [f'{d}/{f}' for d in ls for f in ('current_now', 'current_avg')]
    extra_v = [f'{d}/{f}' for d in ls for f in ('voltage_now', 'voltage_avg')]
    def ok_current(p, v):              # skip chargers/USB that read 0 while unplugged
        return v is not None and abs(int(v)) > 0
    cur = next(((p, v) for p in CUR_CANDIDATES + extra_c for v in [readable(p)] if ok_current(p, v)), None)
    volt = next(((p, v) for p in VOLT_CANDIDATES + extra_v for v in [readable(p)] if v is not None and int(v) > 0), None)
    return cur, volt

class Sampler(threading.Thread):
    """One long-lived adb shell printing current and voltage every 0.2 s (voltage from dumpsys if no file is readable)."""
    def __init__(self, serial, cur_path, volt_path):
        super().__init__(daemon=True); self.samples = []; self.stop = False
        c = f'$(cat /sys/class/power_supply/{cur_path})'
        v = (f'$(cat /sys/class/power_supply/{volt_path})' if volt_path
             else "$(dumpsys battery | grep -m1 ' voltage:' | tr -dc 0-9)")
        loop = f'while true; do echo S {c} {v}; sleep 0.2; done'
        self.p = subprocess.Popen(['adb'] + (['-s', serial] if serial else []) + ['shell', loop],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def run(self):
        for line in self.p.stdout:
            if self.stop: break
            m = re.match(r'S (-?\d+) (\d+)', line.strip())
            if m: self.samples.append((time.time(), to_amps(m.group(1)) * to_volts(m.group(2))))
    def close(self):
        self.stop = True; self.p.terminate()

def window(samples, t0, t1, settle=5.0):
    w = [p for t, p in samples if t0 + settle <= t <= t1]
    return (float(np.median(w)), len(w)) if w else (float('nan'), 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tea_screener_rev1.onnx')
    ap.add_argument('--serial', default=None, help='adb device, e.g. 127.0.0.1:37123 (default: the only one)')
    ap.add_argument('--cycles', type=int, default=3)
    ap.add_argument('--idle', type=float, default=60)
    ap.add_argument('--load', type=float, default=120)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--device', default='')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    print('Step 1  finding a readable battery sensor through adb ...', flush=True)
    cur, volt = discover(a.serial)
    if cur is None:
        r = adb('devices'); print('STOP: no readable battery current file.\n  adb devices:', r.stdout.strip()[:300])
        print('  Use the USB power-meter method instead.'); sys.exit(1)
    dumpsys = adb('shell', 'dumpsys battery', serial=a.serial).stdout
    if volt is None:
        mv = re.search(r' voltage: (\d+)', dumpsys); vstr = mv.group(1) if mv else '3850'
        print('   voltage: no readable file; using dumpsys battery (mV)')
    else:
        vstr = volt[1]
    plugged = [l.strip() for l in dumpsys.splitlines() if re.search(r'(AC|USB|Wireless) powered: true', l)]
    print(f'   current: /sys/class/power_supply/{cur[0]} = {cur[1]} -> {to_amps(cur[1]) * 1000:.0f} mA')
    print(f'   voltage: {volt[0] if volt else "dumpsys battery"} = {vstr} -> {to_volts(vstr):.3f} V | '
          f'power {to_amps(cur[1]) * to_volts(vstr):.2f} W', flush=True)
    if plugged:
        print('STOP: the phone is charging (%s). Unplug it, then run again.' % plugged); sys.exit(1)
    if a.check:
        s = Sampler(a.serial, cur[0], volt[0] if volt else None); s.start(); time.sleep(5); s.close()
        print(f'   sampler: {len(s.samples)} readings in 5 s (expect ~20); median power '
              f'{np.median([p for _, p in s.samples]):.2f} W' if s.samples else '   sampler: no readings')
        print('Check passed: run without --check for the measurement.' if len(s.samples) >= 10 else
              'Too few readings: tell Claude what this printed.'); return

    import onnxruntime as ort
    so = ort.SessionOptions()
    if a.threads > 0: so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
    for _ in range(15): sess.run(None, {name: x})

    s = Sampler(a.serial, cur[0], volt[0] if volt else None); s.start(); time.sleep(3)
    cycles = []
    for c in range(a.cycles):
        print(f'Cycle {c + 1}/{a.cycles}: idle {a.idle:.0f} s ...', flush=True)
        t0 = time.time(); time.sleep(a.idle); t1 = time.time()
        print(f'Cycle {c + 1}/{a.cycles}: model {a.load:.0f} s ...', flush=True)
        n = 0; t2 = time.time()
        while time.time() - t2 < a.load: sess.run(None, {name: x}); n += 1
        t3 = time.time()
        p_idle, ni = window(s.samples, t0, t1); p_load, nl = window(s.samples, t2, t3)
        e = (p_load - p_idle) * (t3 - t2) / n * 1000
        cycles.append(dict(idle_w=round(p_idle, 3), load_w=round(p_load, 3), leaves=n, ms_per_leaf=round((t3 - t2) / n * 1000, 2),
                           mj_per_leaf_above_idle=round(e, 2), samples_idle=ni, samples_load=nl))
        print(f'   idle {p_idle:.2f} W | model {p_load:.2f} W | {n} leaves, {(t3 - t2) / n * 1000:.1f} ms each '
              f'| {e:.1f} mJ per leaf above idle', flush=True)
    s.close()
    e = [c_['mj_per_leaf_above_idle'] for c_ in cycles]
    res = dict(device=a.device, model=os.path.basename(a.model), threads=a.threads, cycles=cycles,
               mj_per_leaf_above_idle=dict(mean=round(float(np.mean(e)), 2), sd=round(float(np.std(e, ddof=1)), 2) if len(e) > 1 else 0.0),
               leaves_per_wh=round(3600 * 1000 / float(np.mean(e)), 0) if np.mean(e) > 0 else None,
               mj_per_leaf_median_idle=(lambda mi: dict(mean=round(float(np.mean([(c_['load_w'] - mi) * c_['ms_per_leaf'] for c_ in cycles])), 2),
                   sd=round(float(np.std([(c_['load_w'] - mi) * c_['ms_per_leaf'] for c_ in cycles], ddof=1)), 2) if len(cycles) > 1 else 0.0,
                   note='load minus median idle over all cycles; robust to one disturbed idle period'))(float(np.median([c_['idle_w'] for c_ in cycles]))),
               current_source=cur[0], voltage_source=(volt[0] if volt else 'dumpsys battery'),
               method='adb shell reading the fuel-gauge current x battery voltage every 0.2 s, '
                      'median over each phase after 5 s settling; phone on battery via wireless debugging')
    print('RESULT %.1f ± %.1f mJ per leaf above idle (%d cycles)' % (res['mj_per_leaf_above_idle']['mean'],
          res['mj_per_leaf_above_idle']['sd'], len(e)))
    print('        %.1f ± %.1f mJ per leaf against the median idle of all cycles (use this if one idle period was disturbed)'
          % (res['mj_per_leaf_median_idle']['mean'], res['mj_per_leaf_median_idle']['sd']))
    print('RESULT ' + json.dumps(res))
    json.dump(res, open('energy_adb_result.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
