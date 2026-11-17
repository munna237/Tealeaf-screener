#!/usr/bin/env python3
"""Energy per inference from the battery charge counter (idle phase vs model phase).

    python energy_test.py --quick      # ~4 min: checks everything works
    python energy_test.py              # ~17 min: the measurement for the paper

Phone: unplugged, airplane mode, screen dimmed (same brightness throughout), other apps closed.
Battery source: /sys/class/power_supply/battery (if readable) or termux-battery-status (Termux:API),
each call limited to 20 s with 3 retries so a frozen call cannot hang the run."""
import argparse, json, os, subprocess, sys, time
import numpy as np

SYS = '/sys/class/power_supply/battery'

def read_sysfs():
    q = int(open(f'{SYS}/charge_counter').read())            # microamp-hours
    v = int(open(f'{SYS}/voltage_now').read())                # microvolts
    return dict(uah=q, volts=v / 1e6, src='sysfs')

def read_api():
    for attempt in range(3):
        try:
            out = subprocess.run(['termux-battery-status'], capture_output=True, text=True, timeout=20).stdout
            d = json.loads(out[out.index('{'):out.rindex('}') + 1])
            v = float(d['voltage']); v = v / 1000 if v > 100 else v
            return dict(uah=int(d['charge_counter']), volts=v, src='termux-api', pct=d.get('percentage'),
                        plugged=d.get('plugged'))
        except Exception as e:
            print(f'   battery read attempt {attempt + 1}/3 failed: {type(e).__name__}', flush=True); time.sleep(2)
    return None

SOURCE = None
def battery():
    global SOURCE
    if SOURCE in (None, 'sysfs'):
        try:
            r = read_sysfs(); SOURCE = 'sysfs'; return r
        except Exception:
            if SOURCE == 'sysfs': return None
    r = read_api()
    if r: SOURCE = 'termux-api'
    return r

def wait(seconds, label):
    t0 = time.time()
    while (left := seconds - (time.time() - t0)) > 0:
        time.sleep(min(30, left)); print(f'   {label}: {min(seconds, time.time() - t0):5.0f} / {seconds} s', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tea_screener_rev1.onnx')
    ap.add_argument('--seconds', type=int, default=480, help='length of EACH phase')
    ap.add_argument('--quick', action='store_true', help='120 s phases, to check the setup')
    a = ap.parse_args()
    secs = 120 if a.quick else a.seconds

    print('Step 1/4  reading the battery ...', flush=True)
    b = battery()
    if not b:
        print('STOP: the battery cannot be read on this phone (sysfs denied and Termux:API not answering).'); sys.exit(1)
    print(f'   OK via {b["src"]}: charge counter {b["uah"]} uAh, {b["volts"]:.3f} V'
          + (f', {b["pct"]}%, {b["plugged"]}' if 'pct' in b else ''), flush=True)
    if b.get('plugged') not in (None, 'UNPLUGGED'):
        print('STOP: unplug the charger first.'); sys.exit(1)

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
    x = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
    for _ in range(15): sess.run(None, {'input': x})

    print(f'Step 2/4  idle phase ({secs} s) - do not touch the phone', flush=True)
    i0 = battery(); wait(secs, 'idle'); i1 = battery()
    print(f'Step 3/4  model phase ({secs} s) - do not touch the phone', flush=True)
    r0 = battery(); t0 = time.time(); n = 0; nxt = 30
    while (el := time.time() - t0) < secs:
        sess.run(None, {'input': x}); n += 1
        if el >= nxt: print(f'   model: {el:5.0f} / {secs} s, {n} leaves', flush=True); nxt += 30
    dur = time.time() - t0; r1 = battery()

    print('Step 4/4  result', flush=True)
    if not all((i0, i1, r0, r1)):
        print('RESULT failed: a battery read failed during the run'); sys.exit(1)
    idle, run = i0['uah'] - i1['uah'], r0['uah'] - r1['uah']
    v = np.mean([i0['volts'], i1['volts'], r0['volts'], r1['volts']])
    res = dict(source=SOURCE, phase_seconds=secs, idle_uah=idle, run_uah=run, inferences=n, run_seconds=round(dur, 1),
               volts=round(float(v), 3), quick=a.quick)
    if run > idle > 0:
        res['mj_per_leaf_above_idle'] = round((run - idle) * 3.6 * v / n, 2)    # uAh * 3.6 = mC ; * V = mJ
        res['mj_per_leaf_total'] = round(run * 3.6 * v / n, 2)
        res['idle_mw'] = round(idle * 3.6 * v / secs, 1); res['run_mw'] = round(run * 3.6 * v / dur, 1)
        print(f"RESULT {res['mj_per_leaf_above_idle']} mJ per leaf above idle "
              f"(idle {res['idle_mw']} mW, running {res['run_mw']} mW, {n} leaves)")
    else:
        print('RESULT the charge counter did not separate idle from load (counter too coarse or did not update)')
    print('RESULT ' + json.dumps(res))
    json.dump(res, open('energy_result_quick.json' if a.quick else 'energy_result.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
