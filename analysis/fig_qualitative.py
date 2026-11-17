"""Fig. (qualitative): our capture vs external datasets vs field photographs, Otsu failure vs SAM.
Tiles are cut from Fig. 1 (dataset_overview.png) and the contact sheets saved by the external and field notebooks."""
import sys
from PIL import Image, ImageDraw, ImageFont
OV, EWU, ALLY, FW, FS, OUT = sys.argv[1:7]
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
T = 300                                            # tile size in the output
def ov_tiles():                                    # Fig. 1: 4 + 3 framed images, label below each
    im = Image.open(OV).convert("RGB"); s = 1.186
    boxes = [(25, 25), (518, 25), (1010, 25), (1503, 25), (272, 577), (765, 577)]
    return [im.crop((int(x * s) + 3, int(y * s) + 3, int((x + 472) * s) - 3, int((y + 472) * s) - 3)).resize((T, T)) for x, y in boxes]
def sheet_tiles(p, n=6, row=0):                    # contact sheets: 3 x 8 tiles of 201 px, title band of 16 px above
    im = Image.open(p).convert("RGB"); xs = [17, 235, 453, 671, 889, 1106, 1324, 1542]; y = [60, 279, 498][row]
    return [im.crop((x, y, x + 201, y + 201)).resize((T, T)) for x in xs[:n]]
rows = [("(a) TeaLeafDiseaseBD (ours): black-cloth backdrop; the Otsu mask isolates the leaf (median 10% of the frame)", ov_tiles()),
        ("(b) External dataset 1: detached leaves, varied backgrounds; crops from the dataset's leaf boxes", sheet_tiles(EWU)),
        ("(c) External dataset 2: light backdrop; the Otsu mask returns the whole frame (median 99.8%), so the frame is scored", sheet_tiles(ALLY)),
        ("(d) Field photographs, four Habiganj estates: whole frame", sheet_tiles(FW)),
        ("(e) The same field set: Segment Anything leaf crop (centre-point prompt; plausible mask on 98.6%)", sheet_tiles(FS))]
G, H, L = 10, 34, 20                               # gap, caption band height, left margin
W = L * 2 + 6 * T + 5 * G
canvas = Image.new("RGB", (W, len(rows) * (H + T + G) + G), "white"); d = ImageDraw.Draw(canvas)
font = ImageFont.truetype(F, 21); fb = ImageFont.truetype(FB, 21)
y = G
for cap, tiles in rows:
    head, rest = cap.split(" ", 1)
    d.text((L, y + 4), head, fill="black", font=fb); d.text((L + d.textlength(head + " ", font=fb), y + 4), rest, fill="#222", font=font)
    y += H
    for i, t in enumerate(tiles): canvas.paste(t, (L + i * (T + G), y))
    y += T + G
canvas.save(OUT, dpi=(300, 300)); print("saved", OUT, canvas.size)
