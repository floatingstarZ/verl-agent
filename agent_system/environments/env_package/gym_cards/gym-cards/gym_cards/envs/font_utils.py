from PIL import ImageFont


def load_font(size, bold=False):
    font_names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "dejavu/DejaVuSans-Bold.ttf" if bold else "dejavu/DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()
