from PIL import Image, ImageDraw, ImageFont
import os
import sys

CHAR_SIZE = 16
MARGIN_RATIO = 0.85
OUTPUT_DIR = "IO/testPhrases"

def load_font(size):
    """Load font with fallbacks - same as training script"""
    for name in [
        "arial.ttf",
        "Arial.ttf"
    ]: # , "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Font Arial (or fallback) not found")

def render_character(ch, img_size):
    """Render a single character - same method as training"""
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    
    # Find the largest font size that fits
    font_size = img_size
    while font_size > 4:
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), ch, font=font)
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        if w <= img_size * MARGIN_RATIO and h <= img_size * MARGIN_RATIO:
            break
        
        font_size -= 1
    
    # Correct centering (considers negative bbox)
    x = (img_size - w) // 2 - bbox[0]
    y = (img_size - h) // 2 - bbox[1]
    
    draw.text((x, y), ch, fill=0, font=font)
    
    return img

def generate_phrase_image(phrase, char_size=CHAR_SIZE, spacing=2):
    """
    Generate a phrase image by compositing individual characters.
    Each character is rendered using the same method as training.
    
    Args:
        phrase: The text to render
        char_size: Size of each character cell (matches training)
        spacing: Pixels between characters
    """
    # Calculate dimensions
    # Space character gets FULL width (to distinguish from inter-char gaps)
    char_widths = []
    for ch in phrase:
        if ch == ' ':
            char_widths.append(char_size)  # Full width for clear space detection
        else:
            char_widths.append(char_size)
    
    total_width = sum(char_widths) + spacing * (len(phrase) - 1)
    height = char_size + 4  # Small vertical padding
    
    # Create the phrase image
    phrase_img = Image.new("L", (total_width, height), 255)
    
    x_offset = 0
    for i, ch in enumerate(phrase):
        if ch == ' ':
            x_offset += char_widths[i] + spacing
            continue
        
        # Render character using same method as training
        char_img = render_character(ch, char_size)
        
        # ===== DISAMBIGUATION HACKS (applied BEFORE pasting) =====
        img_size = char_size
        char_draw = ImageDraw.Draw(char_img)

        cx = img_size // 2
        mid = img_size // 2

        # ---------- I ----------
        if ch == "I":
            char_draw.rectangle((cx - 1, 2, cx + 1, img_size - 3), fill=0)
            char_draw.rectangle((3, 1, img_size - 4, 3), fill=0)
            char_draw.rectangle((3, img_size - 4, img_size - 4, img_size - 2), fill=0)

        # ---------- i (lowercase - make wider for segmentation) ----------
        if ch == "i":
            # wider dot (3 pixels)
            char_draw.rectangle((cx - 1, 2, cx + 1, 4), fill=0)
            # wider stem (3 pixels)
            char_draw.rectangle((cx - 1, 5, cx + 1, img_size - 3), fill=0)

        # ---------- j ----------
        if ch == "j":
            # dot (offset left)
            char_draw.rectangle((cx - 4, 1, cx - 2, 3), fill=0)
            # stem
            char_draw.rectangle((cx - 1, 4, cx + 1, img_size - 4), fill=0)
            # hook bottom right
            char_draw.rectangle((cx + 1, img_size - 4, cx + 4, img_size - 2), fill=0)

        # ---------- t ----------
        if ch == "t":
            # vertical stem shifted left
            char_draw.rectangle((cx - 3, 3, cx - 1, img_size - 3), fill=0)
            # short mid crossbar
            char_draw.rectangle((cx - 5, mid - 1, cx + 1, mid + 1), fill=0)

        # ---------- H ----------
        if ch == "H":
            # left and right stems
            char_draw.rectangle((3, 2, 5, img_size - 3), fill=0)
            char_draw.rectangle((img_size - 6, 2, img_size - 4, img_size - 3), fill=0)
            # thick center bar
            char_draw.rectangle((3, mid - 1, img_size - 4, mid + 1), fill=0)
            # break diagonal symmetry (anti-N cue)
            char_draw.rectangle((cx + 1, mid - 2, cx + 2, mid - 1), fill=255)

        # ---------- N ----------
        if ch == "N":
            # stems
            char_draw.rectangle((3, 2, 5, img_size - 3), fill=0)
            char_draw.rectangle((img_size - 6, 2, img_size - 4, img_size - 3), fill=0)
            # long diagonal (forced)
            for i in range(img_size - 6):
                char_draw.point((5 + i, 2 + i), fill=0)
            # ensure no center bar
            char_draw.rectangle((0, mid - 1, img_size - 1, mid + 1), fill=255)

        # ---------- O (distinguish from 0 and o) ----------
        if ch == "O":
            # O is FULL HEIGHT - extend to edges
            char_draw.ellipse((3, 2, img_size - 4, img_size - 3), outline=0, width=2)

        # ---------- o (distinguish from O and 0) ----------
        if ch == "o":
            # lowercase o is SMALLER and sits in lower half
            char_draw.ellipse((4, 5, img_size - 5, img_size - 3), outline=0, width=2)

        # ---------- 0 (distinguish from O) ----------
        if ch == "0":
            # Add diagonal slash inside zero
            for i in range(img_size - 8):
                char_draw.point((4 + i, img_size - 5 - i), fill=0)
                char_draw.point((5 + i, img_size - 5 - i), fill=0)

        # ---------- 1 (distinguish from l and I) ----------
        if ch == "1":
            # Add base serif and top flag
            char_draw.rectangle((cx - 3, img_size - 4, cx + 3, img_size - 2), fill=0)
            char_draw.rectangle((cx - 3, 2, cx - 1, 4), fill=0)

        # ---------- l (lowercase L - distinguish from 1 and I) ----------
        if ch == "l":
            # curved bottom hook
            char_draw.rectangle((cx, img_size - 5, cx + 3, img_size - 3), fill=0)

        # ---------- k (distinguish from l) ----------
        if ch == "k":
            # vertical stem on left
            char_draw.rectangle((3, 2, 5, img_size - 3), fill=0)
            # diagonal arms going right
            for i in range(5):
                char_draw.point((6 + i, mid - i), fill=0)
                char_draw.point((6 + i, mid + i), fill=0)
                char_draw.point((7 + i, mid - i), fill=0)
                char_draw.point((7 + i, mid + i), fill=0)

        # ---------- Z (uppercase - distinguish from z) ----------
        if ch == "Z":
            # top bar FULL WIDTH
            char_draw.rectangle((2, 2, img_size - 3, 4), fill=0)
            # bottom bar FULL WIDTH
            char_draw.rectangle((2, img_size - 5, img_size - 3, img_size - 3), fill=0)
            # diagonal
            for i in range(img_size - 6):
                char_draw.point((img_size - 4 - i, 4 + i), fill=0)
                char_draw.point((img_size - 5 - i, 4 + i), fill=0)

        # ---------- z (lowercase - distinguish from Z) ----------
        if ch == "z":
            # smaller, starts lower
            char_draw.rectangle((4, mid - 2, img_size - 5, mid), fill=0)
            char_draw.rectangle((4, img_size - 5, img_size - 5, img_size - 3), fill=0)
            for i in range(6):
                char_draw.point((img_size - 6 - i, mid + i), fill=0)

        # ---------- X (uppercase - distinguish from x) ----------
        if ch == "X":
            # FULL HEIGHT diagonals
            for i in range(img_size - 4):
                char_draw.point((2 + i, 2 + i), fill=0)
                char_draw.point((3 + i, 2 + i), fill=0)
                char_draw.point((img_size - 3 - i, 2 + i), fill=0)
                char_draw.point((img_size - 4 - i, 2 + i), fill=0)

        # ---------- x (lowercase - distinguish from X) ----------
        if ch == "x":
            # SMALLER x - starts lower, shorter height
            start_y = mid - 2
            for i in range(7):
                char_draw.point((4 + i, start_y + i), fill=0)
                char_draw.point((5 + i, start_y + i), fill=0)
                char_draw.point((img_size - 5 - i, start_y + i), fill=0)
                char_draw.point((img_size - 6 - i, start_y + i), fill=0)

        # ---------- y (distinguish from 7) ----------
        if ch == "y":
            # V shape at top, then descender
            for i in range(5):
                char_draw.point((4 + i, mid - 3 + i), fill=0)
                char_draw.point((img_size - 5 - i, mid - 3 + i), fill=0)
            # descender going down and LEFT
            char_draw.rectangle((cx - 2, mid + 2, cx, img_size - 2), fill=0)

        # ---------- 7 (distinguish from y) ----------
        if ch == "7":
            # strong top bar
            char_draw.rectangle((3, 2, img_size - 4, 4), fill=0)
            # diagonal going DOWN-LEFT (no descender below baseline)
            for i in range(img_size - 6):
                char_draw.point((img_size - 5 - i, 4 + i), fill=0)
                char_draw.point((img_size - 6 - i, 4 + i), fill=0)

        # ---------- b (distinguish from d) ----------
        if ch == "b":
            char_draw.rectangle((3, 2, 5, img_size - 3), fill=0)

        # ---------- d (distinguish from b) ----------
        if ch == "d":
            char_draw.rectangle((img_size - 6, 2, img_size - 4, img_size - 3), fill=0)

        # ---------- T (distinguish from f) ----------
        if ch == "T":
            char_draw.rectangle((2, 2, img_size - 3, 4), fill=0)
            char_draw.rectangle((cx - 1, 4, cx + 1, img_size - 3), fill=0)
        
        # Paste centered vertically (AFTER hacks)
        y_offset = (height - char_size) // 2
        phrase_img.paste(char_img, (x_offset, y_offset))
        
        x_offset += char_widths[i] + spacing
    
    # Binarize the phrase image to remove anti-aliasing
    # This ensures the phrase pixels match the training data (pure 0 or 255)
    phrase_img = phrase_img.point(lambda x: 0 if x < 128 else 255)
    
    return phrase_img

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test phrases to generate - 16x16 for alphanumeric
    randomS = "".join(__import__('random').choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=50))
    phrases_alpha = [
        ("HELLO", "hello_train.png", 16),
        ("WORLD", "world_train.png", 16),
        ("Test123", "mixed_train.png", 16),
        ("HELLO WORLD", "hello_world_train.png", 16),
        ("ABC abc 123", "full_test_train.png", 16),
        ("AEDproject", "aed_train.png", 16),
        ("NeLsOn e Eng RaFeIrO", "nelraf.png", 16),
        ("a1b2C3D4f5C 6 7 fF gG9HjitI HNijk", "crazy1.png", 16),
        ("CrAzY xYz 987 QwErTy", "crazy_train.png", 16),
        (randomS, "crazy2.png", 16),
    ]
    
    # Digit phrases - 8x8 for digit model
    phrases_digits = [
        ("12345", "digits_train.png", 8),
        ("67890", "digits2_train.png", 8),
        ("0123456789", "all_digits_train.png", 8),
    ]
    
    all_phrases = phrases_alpha + phrases_digits
    
    # Also accept command line arguments
    if len(sys.argv) > 1:
        custom_phrase = " ".join(sys.argv[1:])
        safe_name = custom_phrase.replace(" ", "_")[:20] + "_train.png"
        # Default to 16x16 for custom phrases
        all_phrases = [(custom_phrase, safe_name, 16)]
    
    for phrase, filename, char_size in all_phrases:
        img = generate_phrase_image(phrase, char_size=char_size)
        filepath = os.path.join(OUTPUT_DIR, filename)
        img.save(filepath)
        print(f"Generated: {filepath} ({img.width}x{img.height}, {char_size}x{char_size} chars) - '{phrase}'")
    
    print(f"\nGenerated {len(all_phrases)} phrase images in '{OUTPUT_DIR}/'")
    print("\nUsage:")
    print("  ./testDriverPhrase IO/testPhrases/hello_train.png")
    print("  ./testDriverPhrase IO/testPhrases/digits_train.png digits")

if __name__ == "__main__":
    main()