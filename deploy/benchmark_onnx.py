#!/usr/bin/env python3
"""On-device benchmark for the deployed screener (Reviewer 1.6: memory, energy, device spread).

Runs on any machine with onnxruntime, including Android under Termux:
    pkg install python && pip install numpy onnxruntime
    python benchmark_onnx.py --model tea_screener.onnx --threads 1
    python benchmark_onnx.py --model tea_screener.onnx --threads 0      # 0 = all cores

Reports: model load time, latency median / p95 over --runs timed inferences after --warmup,
peak resident memory of the process (and the increase caused by loading + running the model),
and, if the kernel exposes battery current and voltage, an approximate energy per inference.
The energy figure is a whole-device estimate (screen and background included), so run it with the
screen dimmed, airplane mode on, and the phone unplugged, and report it as approximate."""
import argparse, json, os, platform, resource, sys, threading, time
import numpy as np

def rss_mb():                      # current resident set size
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float('nan')

def peak_rss_mb():                 # ru_maxrss is KiB on Linux/Android, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == 'darwin' else r / 1024

BATT = '/sys/class/power_supply/battery'
NOMINAL_V = 3.85          # used only when the battery voltage cannot be read
POWER_SOURCE = None       # filled in by read_power_w(): 'sysfs' or 'termux-api'
POWER_ERRORS = {}         # why each source failed, reported if no source works
SCRIPT_VERSION = 'v3 (current_average + measured voltage)'
CURRENT_FIELD, VOLT_SOURCE = None, None

def _sysfs_power():
    i = abs(int(open(f'{BATT}/current_now').read()))   # microamps
    v = int(open(f'{BATT}/voltage_now').read())         # microvolts
    return i * v / 1e12

def _termux_api_power():
    """Termux:API fallback (app 'Termux:API' + `pkg install termux-api`).
    Phones report either 'current' or 'current_average' (usually microamps) and often 'voltage' (millivolts)."""
    import subprocess
    global CURRENT_FIELD, VOLT_SOURCE
    out = subprocess.run(['termux-battery-status'], capture_output=True, text=True, timeout=30).stdout
    d = json.loads(out[out.index('{'):out.rindex('}') + 1])     # tolerate any text around the JSON
    for key in ('current', 'current_now', 'current_average'):
        if d.get(key) not in (None, 0):
            CURRENT_FIELD = key; cur = abs(float(d[key])); break
    else:
        return None
    amps = cur / 1e6 if cur > 20000 else cur / 1e3          # microamps on most devices, milliamps on some
    v = d.get('voltage')
    if v:
        v = float(v); volts = v / 1000 if v > 100 else v; VOLT_SOURCE = 'measured'
    else:
        volts = NOMINAL_V; VOLT_SOURCE = 'nominal 3.85 V'
    return amps * volts

def read_power_w():
    """Instantaneous whole-device power in watts, or None if no source is readable."""
    global POWER_SOURCE
    for name, fn in (('sysfs', _sysfs_power), ('termux-api', _termux_api_power)):
        if POWER_SOURCE not in (None, name): continue
        try:
            p = fn()
            if p is not None:
                POWER_SOURCE = name; return p
            POWER_ERRORS[name] = 'returned no current value'
        except Exception as e:
            POWER_ERRORS[name] = '%s: %s' % (type(e).__name__, str(e)[:120])
    return None

class PowerSampler(threading.Thread):
    def __init__(self, period=0.05):
        period = period if POWER_SOURCE == 'sysfs' else 0.5   # termux-battery-status is slow to call
        super().__init__(daemon=True); self.period, self.samples, self.stop_flag = period, [], False
    def run(self):
        while not self.stop_flag:
            p = read_power_w()
            if p is not None: self.samples.append(p)
            time.sleep(self.period)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tea_screener.onnx')
    ap.add_argument('--threads', type=int, default=1, help='intra-op threads; 0 = runtime default (all cores)')
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=15)
    ap.add_argument('--idle_s', type=float, default=45.0, help='idle-power baseline duration (energy only)')
    ap.add_argument('--energy_runs', type=int, default=3000)
    ap.add_argument('--device', default='', help='free-text device name for the report')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import onnxruntime as ort
    rss0 = rss_mb()
    so = ort.SessionOptions()
    if a.threads > 0:
        so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    t = time.perf_counter()
    sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
    load_ms = (time.perf_counter() - t) * 1000
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    for _ in range(a.warmup): sess.run(None, {inp.name: x})
    lat = []
    for _ in range(a.runs):
        t = time.perf_counter(); sess.run(None, {inp.name: x}); lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    rep = dict(script=SCRIPT_VERSION, device=a.device or platform.node(), machine=platform.machine(), python=platform.python_version(),
               onnxruntime=ort.__version__, model=os.path.basename(a.model),
               model_mb=os.path.getsize(a.model) / 1e6, input_shape=shape,
               threads=a.threads or 'default', load_ms=round(load_ms, 1),
               latency_ms=dict(median=round(float(np.median(lat)), 2), p95=round(float(np.percentile(lat, 95)), 2),
                               mean=round(float(lat.mean()), 2), runs=a.runs, warmup=a.warmup),
               fps_median=round(1000 / float(np.median(lat)), 1),
               rss_before_mb=round(rss0, 1), rss_after_mb=round(rss_mb(), 1),
               peak_rss_mb=round(peak_rss_mb(), 1), model_rss_increase_mb=round(rss_mb() - rss0, 1))

    if read_power_w() is None:
        rep['energy'] = dict(status='battery current not readable', reasons=POWER_ERRORS,
                             fix='install the Termux:API app and run `pkg install termux-api`, or use an external USB meter')
    else:
        s = PowerSampler(); s.start()
        time.sleep(a.idle_s); idle = list(s.samples)[len(s.samples) // 3:]      # drop the first third (settling)
        s.samples.clear()
        t = time.perf_counter()
        for _ in range(a.energy_runs): sess.run(None, {inp.name: x})
        dur = time.perf_counter() - t; busy = list(s.samples); s.stop_flag = True
        busy = busy[len(busy) // 3:]                                              # drop the ramp-up
        if idle and busy:
            p_idle, p_busy = float(np.median(idle)), float(np.median(busy))
            rep['energy'] = dict(idle_w=round(p_idle, 3), busy_w=round(p_busy, 3),
                                 mj_per_inference_above_idle=round((p_busy - p_idle) * dur / a.energy_runs * 1000, 3),
                                 mj_per_inference_total=round(p_busy * dur / a.energy_runs * 1000, 3),
                                 busy_seconds=round(dur, 1), source=POWER_SOURCE, current_field=CURRENT_FIELD,
                                 voltage=VOLT_SOURCE, samples_idle=len(idle), samples_busy=len(busy),
                                 note='whole-device estimate from battery current x voltage; approximate')
        else:
            rep['energy'] = 'too few power samples; increase --idle_s / --energy_runs'
    print(json.dumps(rep, indent=2))
    if a.out: json.dump(rep, open(a.out, 'w'), indent=2)

if __name__ == '__main__':
    main()
