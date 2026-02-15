from PIL import Image, ImageDraw, ImageFont
import string
import os

## Thank you, Chat GPT. I did not have the patience to do this.
## I only tuned this.

IMG_SIZE = 16
MARGIN_RATIO = 0.85
OUTPUT_DIR = "IO/pngAlphaChars"

CHARS = (
    string.ascii_uppercase +
    string.ascii_lowercase +
    string.digits
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_font(size):
    for name in [
        "arial.ttf",
        "Arial.ttf"
    ]: #, "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Font not found (Arial)")

for ch in CHARS:
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)

    # find max font size
    font_size = IMG_SIZE
    while font_size > 4:
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), ch, font=font)

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= IMG_SIZE * MARGIN_RATIO and h <= IMG_SIZE * MARGIN_RATIO:
            break

        font_size -= 1

    # center text (bbox may be negative)
    x = (IMG_SIZE - w) // 2 - bbox[0]
    y = (IMG_SIZE - h) // 2 - bbox[1]

    draw.text((x, y), ch, fill=0, font=font)

    # HACKS 

    cx = IMG_SIZE // 2
    mid = IMG_SIZE // 2

    # ---------- I ----------
    if ch == "I":
        draw.rectangle((cx - 1, 2, cx + 1, IMG_SIZE - 3), fill=0)
        draw.rectangle((3, 1, IMG_SIZE - 4, 3), fill=0)
        draw.rectangle((3, IMG_SIZE - 4, IMG_SIZE - 4, IMG_SIZE - 2), fill=0)

    # ---------- i (lowercase - make wider for segmentation) ----------
    if ch == "i":
        # wider dot (3 pixels)
        draw.rectangle((cx - 1, 2, cx + 1, 4), fill=0)
        # wider stem (3 pixels)
        draw.rectangle((cx - 1, 5, cx + 1, IMG_SIZE - 3), fill=0)

    # ---------- j ----------
    if ch == "j":
        # dot (offset left)
        draw.rectangle((cx - 4, 1, cx - 2, 3), fill=0)
        # stem
        draw.rectangle((cx - 1, 4, cx + 1, IMG_SIZE - 4), fill=0)
        # hook bottom right
        draw.rectangle((cx + 1, IMG_SIZE - 4, cx + 4, IMG_SIZE - 2), fill=0)

    # ---------- t ----------
    if ch == "t":
        # vertical stem shifted left
        draw.rectangle((cx - 3, 3, cx - 1, IMG_SIZE - 3), fill=0)
        # short mid crossbar
        draw.rectangle((cx - 5, mid - 1, cx + 1, mid + 1), fill=0)

    # ---------- H ----------
    if ch == "H":
        # left and right stems
        draw.rectangle((3, 2, 5, IMG_SIZE - 3), fill=0)
        draw.rectangle((IMG_SIZE - 6, 2, IMG_SIZE - 4, IMG_SIZE - 3), fill=0)
        # thick center bar
        draw.rectangle((3, mid - 1, IMG_SIZE - 4, mid + 1), fill=0)
        # break diagonal symmetry (anti-N cue)
        draw.rectangle((cx + 1, mid - 2, cx + 2, mid - 1), fill=255)

    # ---------- N ----------
    if ch == "N":
        # stems
        draw.rectangle((3, 2, 5, IMG_SIZE - 3), fill=0)
        draw.rectangle((IMG_SIZE - 6, 2, IMG_SIZE - 4, IMG_SIZE - 3), fill=0)
        # long diagonal (forced)
        for i in range(IMG_SIZE - 6):
            draw.point((5 + i, 2 + i), fill=0)
        # ensure no center bar
        draw.rectangle((0, mid - 1, IMG_SIZE - 1, mid + 1), fill=255)

    # ---------- O (distinguish from 0 and o) ----------
    if ch == "O":
        # O is FULL HEIGHT - extend to edges
        draw.ellipse((3, 2, IMG_SIZE - 4, IMG_SIZE - 3), outline=0, width=2)

    # ---------- o (distinguish from O and 0) ----------
    if ch == "o":
        # lowercase o is SMALLER and sits in lower half
        draw.ellipse((4, 5, IMG_SIZE - 5, IMG_SIZE - 3), outline=0, width=2)

    # ---------- 0 (distinguish from O) ----------
    if ch == "0":
        # Add diagonal slash inside zero
        for i in range(IMG_SIZE - 8):
            draw.point((4 + i, IMG_SIZE - 5 - i), fill=0)
            draw.point((5 + i, IMG_SIZE - 5 - i), fill=0)

    # ---------- 1 (distinguish from l and I) ----------
    if ch == "1":
        # Add base serif and top flag
        draw.rectangle((cx - 3, IMG_SIZE - 4, cx + 3, IMG_SIZE - 2), fill=0)  # base
        draw.rectangle((cx - 3, 2, cx - 1, 4), fill=0)  # top left flag

    # ---------- l (lowercase L - distinguish from 1 and I) ----------
    if ch == "l":
        # curved bottom hook
        draw.rectangle((cx, IMG_SIZE - 5, cx + 3, IMG_SIZE - 3), fill=0)

    # ---------- k (distinguish from l) ----------
    if ch == "k":
        # vertical stem on left
        draw.rectangle((3, 2, 5, IMG_SIZE - 3), fill=0)
        # diagonal arms going right
        for i in range(5):
            draw.point((6 + i, mid - i), fill=0)
            draw.point((6 + i, mid + i), fill=0)
            draw.point((7 + i, mid - i), fill=0)
            draw.point((7 + i, mid + i), fill=0)

    # ---------- Z (uppercase - distinguish from z) ----------
    if ch == "Z":
        # top bar FULL WIDTH
        draw.rectangle((2, 2, IMG_SIZE - 3, 4), fill=0)
        # bottom bar FULL WIDTH
        draw.rectangle((2, IMG_SIZE - 5, IMG_SIZE - 3, IMG_SIZE - 3), fill=0)
        # diagonal
        for i in range(IMG_SIZE - 6):
            draw.point((IMG_SIZE - 4 - i, 4 + i), fill=0)
            draw.point((IMG_SIZE - 5 - i, 4 + i), fill=0)

    # ---------- z (lowercase - distinguish from Z) ----------
    if ch == "z":
        # smaller, starts lower
        draw.rectangle((4, mid - 2, IMG_SIZE - 5, mid), fill=0)  # top bar
        draw.rectangle((4, IMG_SIZE - 5, IMG_SIZE - 5, IMG_SIZE - 3), fill=0)  # bottom bar
        for i in range(6):
            draw.point((IMG_SIZE - 6 - i, mid + i), fill=0)

    # ---------- y (distinguish from 7) ----------
    if ch == "y":
        # V shape at top, then descender
        for i in range(5):
            draw.point((4 + i, mid - 3 + i), fill=0)
            draw.point((IMG_SIZE - 5 - i, mid - 3 + i), fill=0)
        # descender going down and LEFT
        draw.rectangle((cx - 2, mid + 2, cx, IMG_SIZE - 2), fill=0)

    # ---------- 7 (distinguish from y) ----------
    if ch == "7":
        # strong top bar
        draw.rectangle((3, 2, IMG_SIZE - 4, 4), fill=0)
        # diagonal going DOWN-LEFT (no descender below baseline)
        for i in range(IMG_SIZE - 6):
            draw.point((IMG_SIZE - 5 - i, 4 + i), fill=0)
            draw.point((IMG_SIZE - 6 - i, 4 + i), fill=0)

    # ---------- b (distinguish from d) ----------
    if ch == "b":
        # ensure stem on LEFT, bowl on right
        draw.rectangle((3, 2, 5, IMG_SIZE - 3), fill=0)  # left stem

    # ---------- d (distinguish from b) ----------
    if ch == "d":
        # ensure stem on RIGHT, bowl on left
        draw.rectangle((IMG_SIZE - 6, 2, IMG_SIZE - 4, IMG_SIZE - 3), fill=0)  # right stem

    # ---------- X (uppercase - distinguish from x) ----------
    if ch == "X":
        # FULL HEIGHT diagonals
        for i in range(IMG_SIZE - 4):
            draw.point((2 + i, 2 + i), fill=0)
            draw.point((3 + i, 2 + i), fill=0)
            draw.point((IMG_SIZE - 3 - i, 2 + i), fill=0)
            draw.point((IMG_SIZE - 4 - i, 2 + i), fill=0)

    # ---------- x (lowercase - distinguish from X) ----------
    if ch == "x":
        # SMALLER x - starts lower, shorter height
        start_y = mid - 2
        for i in range(7):
            draw.point((4 + i, start_y + i), fill=0)
            draw.point((5 + i, start_y + i), fill=0)
            draw.point((IMG_SIZE - 5 - i, start_y + i), fill=0)
            draw.point((IMG_SIZE - 6 - i, start_y + i), fill=0)

    # ---------- T (distinguish from f) ----------
    if ch == "T":
        # Strong top bar, centered stem
        draw.rectangle((2, 2, IMG_SIZE - 3, 4), fill=0)  # wide top bar
        draw.rectangle((cx - 1, 4, cx + 1, IMG_SIZE - 3), fill=0)  # center stem

    # ================= SAVE =================

    filename = f"{ord(ch):03d}_{ch}.png"
    img.save(os.path.join(OUTPUT_DIR, filename))

print(f"Generated {len(CHARS)} PNGs in '{OUTPUT_DIR}/'")