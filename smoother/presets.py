from fractions import Fraction


CODEC_PRESETS = {"h264": "playback", "prores": "editing", "hevc": "apple-playback"}
PRESET_CODECS = {preset: codec for codec, preset in CODEC_PRESETS.items()} | {"share": "h264"}


def export_dimensions(media, preset):
    width, height = media["width"], media["height"]
    if preset != "share":
        return width, height
    display_width = width * Fraction(media["sar"])
    bounds = (1920, 1080) if display_width >= height else (1080, 1920)
    scale = min(1, bounds[0] / display_width, Fraction(bounds[1], height))
    return tuple(max(2, 2 * int(edge * scale / 2)) for edge in (display_width, height))
