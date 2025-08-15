## Krishna-themed Baby Photo Composer

This tool adds a Krishna theme (crown with peacock feather, tilak, garland, flute and butter pot) to a baby photo while keeping the face unchanged. It does NOT alter facial features; it only overlays themed elements.

### Setup

```bash
python -m pip install -r requirements.txt
```

### Usage

Place your photo somewhere accessible, then run:

```bash
python krishna_edit.py /absolute/path/to/your_photo.jpg -o /absolute/path/to/output_krishna.png
```

If face detection struggles, you can disable it and rely on a center heuristic:

```bash
python krishna_edit.py /absolute/path/to/your_photo.jpg -o /absolute/path/to/output_krishna.png --no-face-detect
```

Notes:
- Output is saved as PNG to preserve transparency and colors.
- The script draws overlays programmatically, so no external asset downloads are needed.
- You can run it multiple times and choose the best result.