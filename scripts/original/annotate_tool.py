"""
annotate_tool.py -- inline lesion box annotator for Colab.

Usage (in Colab):
    import annotate_tool as at
    at.annotate(sample, save_path)      # sample = list of dicts with 'path','class'

Draw a box by dragging over each lesion. Multiple boxes per image allowed.
Buttons: Undo (remove last box), Skip (no visible lesion), Next (save + advance).
Progress autosaves to `save_path` (JSON on Drive) after every image, so a
disconnect never loses work -- re-running resumes where you stopped.

Boxes are stored NORMALIZED (x0,y0,x1,y1 in [0,1]) so they are independent of
display size and can be mapped onto any attention grid.
"""

import os, json, base64, io
from PIL import Image


def _load_existing(save_path):
    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return {}


def annotate(sample, save_path, max_display=560):
    """
    sample    : list of {'path': str, 'class': str}
    save_path : JSON file on Drive; resumes automatically if it exists.
    """
    from IPython.display import display, HTML, Javascript
    from google.colab import output as colab_output

    done = _load_existing(save_path)
    todo = [s for s in sample if s["path"] not in done]
    print(f"Annotation set: {len(sample)} images | done: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("All images annotated. Run the evaluation cell next.")
        return done

    state = {"done": done}

    def _save(path, boxes, cls):
        state["done"][path] = {"boxes": boxes, "class": cls}
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(state["done"], f, indent=2)
        return {"saved": len(state["done"])}

    colab_output.register_callback("notebook.save_boxes", _save)

    # embed the remaining images as base64 so JS can page through without re-calling python
    payload = []
    for s in todo:
        im = Image.open(s["path"]).convert("RGB")
        w, h = im.size
        scale = min(max_display / w, max_display / h, 1.0)
        disp = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO(); disp.save(buf, format="JPEG", quality=85)
        payload.append({
            "path": s["path"],
            "cls": s["class"],
            "b64": base64.b64encode(buf.getvalue()).decode(),
            "w": disp.size[0], "h": disp.size[1],
        })

    html = """
<style>
  #annwrap { font-family: system-ui, sans-serif; }
  #cv { border:2px solid #333; cursor: crosshair; display:block; }
  .btn { padding:8px 16px; margin:6px 6px 0 0; font-size:14px; cursor:pointer;
         border-radius:6px; border:1px solid #999; background:#f5f5f5; }
  .btn:hover { background:#e5e5e5; }
  #next { background:#2ecc71; color:#fff; border-color:#27ae60; font-weight:600; }
  #skip { background:#f39c12; color:#fff; border-color:#d68910; }
  #hdr { font-size:15px; margin-bottom:6px; }
  #cnt { color:#666; font-size:13px; }
</style>
<div id="annwrap">
  <div id="hdr"><b id="cls"></b> &nbsp; <span id="cnt"></span></div>
  <canvas id="cv"></canvas>
  <div>
    <button class="btn" id="undo">Undo box</button>
    <button class="btn" id="skip">No visible lesion (skip)</button>
    <button class="btn" id="next">Save &amp; Next &rarr;</button>
  </div>
  <div id="cnt2" style="margin-top:6px;color:#666;font-size:13px;">
    Drag to draw a box around each lesion. Multiple boxes allowed.
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
let i = 0, boxes = [], drag = null;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const img = new Image();

function load() {
  if (i >= DATA.length) {
    document.getElementById('annwrap').innerHTML =
      "<h3>Done — all images annotated. Run the evaluation cell next.</h3>";
    return;
  }
  const d = DATA[i];
  boxes = [];
  cv.width = d.w; cv.height = d.h;
  document.getElementById('cls').textContent = d.cls;
  document.getElementById('cnt').textContent = `image ${i+1} / ${DATA.length}`;
  img.onload = draw;
  img.src = 'data:image/jpeg;base64,' + d.b64;
}
function draw() {
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  ctx.lineWidth = 3; ctx.strokeStyle = '#e74c3c';
  boxes.forEach(b => ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]));
  if (drag) { ctx.strokeStyle='#3498db';
    ctx.strokeRect(drag[0],drag[1],drag[2]-drag[0],drag[3]-drag[1]); }
}
cv.onmousedown = e => { const r=cv.getBoundingClientRect();
  drag=[e.clientX-r.left, e.clientY-r.top, e.clientX-r.left, e.clientY-r.top]; };
cv.onmousemove = e => { if(!drag) return; const r=cv.getBoundingClientRect();
  drag[2]=e.clientX-r.left; drag[3]=e.clientY-r.top; draw(); };
cv.onmouseup = e => {
  if(!drag) return;
  const w=Math.abs(drag[2]-drag[0]), h=Math.abs(drag[3]-drag[1]);
  if (w>5 && h>5) boxes.push([Math.min(drag[0],drag[2]), Math.min(drag[1],drag[3]),
                              Math.max(drag[0],drag[2]), Math.max(drag[1],drag[3])]);
  drag=null; draw();
};
document.getElementById('undo').onclick = () => { boxes.pop(); draw(); };
function commit(bs) {
  const d = DATA[i];
  const norm = bs.map(b => [b[0]/d.w, b[1]/d.h, b[2]/d.w, b[3]/d.h]);
  google.colab.kernel.invokeFunction('notebook.save_boxes', [d.path, norm, d.cls], {});
  i++; load();
}
document.getElementById('skip').onclick = () => commit([]);
document.getElementById('next').onclick = () => {
  if (boxes.length === 0) { alert('Draw at least one box, or press Skip.'); return; }
  commit(boxes);
};
load();
</script>
"""
    display(HTML(html.replace("__PAYLOAD__", json.dumps(payload))))
    return state["done"]
