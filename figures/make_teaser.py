"""Compose the Crossroads teaser figure from UI captures in figures/captures/.

The captures are screenshots of Crossroads running on examples/swe (SWE-bench runs only),
taken in a 1440x900 browser window at 2x scale.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

S = 1.44  # captures are 2880 px wide; crop boxes below are in 2000-px display units
def crop(path, box):
    im = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    return im.crop((round(x0 * S), round(y0 * S), round(x1 * S), round(y1 * S)))
def fit_h(im, h): return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
def fit_w(im, w): return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)

A, D, F = str(HERE / "captures" / "a_overview.png"), str(HERE / "captures" / "b_detail.png"), str(HERE / "captures" / "c_failure.png")
pa = crop(A, (25, 95, 990, 605))
pb = crop(A, (1032, 195, 1955, 935))
pc = crop(F, (20, 90, 1985, 665))
pd = crop(D, (20, 428, 1265, 1022))
pe = crop(D, (1480, 330, 1990, 490))
pt = crop(D, (1480, 228, 1990, 300))

W, GAP, LAB, PAD = 3600, 56, 92, 60
def load_font(size, bold):
    """Return Helvetica on macOS, DejaVu Sans elsewhere, or PIL's default font."""
    for path, index in (("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


font, font_r = load_font(50, True), load_font(46, False)
inner = W - 2 * PAD

# Row 1: a and b at equal height
ra = pa.width / pa.height; rb = pb.width / pb.height
h1 = round((inner - GAP) / (ra + rb)); a1, b1 = fit_h(pa, h1), fit_h(pb, h1)
# Row 2: c full width
c2 = fit_w(pc, inner)
# Row 3: d left, e stacked at the top of the right column
k = inner / pc.width
def scale(im, k): return im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
d3 = scale(pd, k)
right_w = inner - d3.width - GAP
ve, vt = fit_w(pe, right_w), fit_w(pt, right_w)
e3 = Image.new("RGB", (right_w, ve.height + vt.height + 24), "white")
e3.paste(ve, (0, 0)); e3.paste(vt, (0, ve.height + 24))
h3 = max(d3.height, e3.height)
rows = [h1, c2.height, h3]
H = PAD + sum(LAB + r for r in rows) + 2 * GAP + PAD
fig = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(fig)

def place(im, x, y, letter, title):
    dr.text((x, y), letter, font=font, fill="#111827")
    dr.text((x + 52, y + 4), title, font=font_r, fill="#374151")
    fig.paste(im, (x, y + LAB))
    dr.rectangle((x - 1, y + LAB - 1, x + im.width, y + LAB + im.height), outline="#D1D5DB", width=3)

y = PAD
place(a1, PAD, y, "a", "Corpus overview of five SWE-bench runs")
place(b1, PAD + a1.width + GAP, y, "b", "Agent thoughts grouped by topic")
y += LAB + h1 + GAP
place(c2, PAD, y, "c", "Failure analysis: heuristic failure modes and LLM-judge patterns")
y += LAB + c2.height + GAP
place(d3, PAD, y, "d", "Step trace with observation compression")
place(e3, PAD + d3.width + GAP, y, "e", "Failure verdict for this run")

fig.save(str(HERE / "crossroads_teaser.png"), dpi=(300, 300))
fig.save(str(HERE / "crossroads_teaser.pdf"), resolution=300)
print(fig.size)
