from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


DEFAULT_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]

BRANDS = [
    "ЛЕНТА",
    "PREMIUM CLUB",
    "365 ДНЕЙ",
    "СЕМЕЙНЫЕ СЕКРЕТЫ",
    "ЗОЛОТАЯ ДОЛИНА",
    "РУССКИЙ ПРОДУКТ",
    "ВКУСНЫЙ ДОМ",
    "БАБУШКИН ПОГРЕБОК",
    "GREEN RAY",
    "FINE LIFE",
]

PRODUCTS = [
    "мед",
    "мед цветочный",
    "мед липовый",
    "мед гречишный",
    "варенье",
    "джем",
    "конфитюр",
    "сироп",
    "паста томатная",
    "соус",
    "кетчуп",
    "горчица",
    "икра овощная",
    "лечо",
    "огурцы маринованные",
    "томаты в собственном соку",
    "горошек зеленый",
    "кукуруза сладкая",
    "фасоль красная",
    "оливки без косточки",
    "маслины",
    "молоко",
    "сметана",
    "творог",
    "йогурт",
    "сыр плавленый",
    "масло сливочное",
    "чай черный",
    "кофе растворимый",
    "печенье",
    "шоколад",
    "хлопья овсяные",
]

FLAVORS = [
    "лесные ягоды",
    "клубника",
    "малина",
    "абрикос",
    "черника",
    "вишня",
    "яблоко",
    "апельсин",
    "натуральный",
    "классический",
    "пряный",
    "острый",
    "сливочный",
    "домашний",
]

COUNTRIES = [
    "Россия",
    "Беларусь",
    "Казахстан",
    "Армения",
    "Сербия",
    "Турция",
    "Италия",
    "Испания",
]

UNITS = ["г", "кг", "мл", "л"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic PaddleOCR text recognition dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4315)
    parser.add_argument("--max-label-length", type=int, default=40)
    parser.add_argument("--preview-count", type=int, default=80)
    return parser.parse_args()


def load_allowed_chars(dict_path: Path) -> set[str]:
    chars = {line.rstrip("\n") for line in dict_path.read_text(encoding="utf-8").splitlines()}
    chars.add(" ")
    return {char for char in chars if len(char) == 1}


def available_fonts() -> list[Path]:
    fonts = [Path(path) for path in DEFAULT_FONTS if Path(path).exists()]
    if not fonts:
        raise RuntimeError("No Windows fonts found for synthetic OCR rendering")
    return fonts


def random_weight(rng: random.Random) -> str:
    unit = rng.choice(UNITS)
    if unit == "кг":
        value = rng.choice(["0,2", "0,25", "0,3", "0,4", "0,5", "0,8", "1"])
    elif unit == "л":
        value = rng.choice(["0,25", "0,33", "0,5", "0,75", "1", "1,5"])
    elif unit == "мл":
        value = str(rng.choice([90, 100, 180, 200, 250, 330, 500, 750, 900]))
    else:
        value = str(rng.choice([80, 100, 125, 180, 200, 250, 270, 300, 350, 400, 450, 500, 700, 900]))
    return f"{value}{unit}" if rng.random() < 0.55 else f"{value} {unit}"


def clean_label(text: str, allowed: set[str]) -> str:
    text = " ".join(text.replace("\t", " ").replace("\n", " ").split())
    text = "".join(char for char in text if char in allowed)
    return " ".join(text.split())


def sample_label(rng: random.Random, allowed: set[str], max_len: int) -> str:
    for _ in range(200):
        product = rng.choice(PRODUCTS)
        flavor = rng.choice(FLAVORS)
        brand = rng.choice(BRANDS)
        country = rng.choice(COUNTRIES)
        weight = random_weight(rng)
        variant = rng.randrange(9)
        if variant == 0:
            text = f"{product} {flavor} {weight}"
        elif variant == 1:
            text = f"{brand} {product} {weight}"
        elif variant == 2:
            text = f"{product.upper()} {flavor} ({country})"
        elif variant == 3:
            text = f"{product} {brand} {weight}"
        elif variant == 4:
            text = f"{brand} {flavor} {weight}"
        elif variant == 5:
            text = f"{product} {flavor} {country}"
        elif variant == 6:
            text = f"{product} {weight} {country}"
        elif variant == 7:
            text = f"{brand} {product.upper()}"
        else:
            text = f"{product} {flavor}"

        if rng.random() < 0.22:
            text = text.upper()
        if rng.random() < 0.12:
            text = text.replace(" ", "  ", 1)
        text = clean_label(text, allowed)
        if 2 <= len(text) <= max_len:
            return text
    return "мед цветочный 500г"


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(image).astype(np.int16)
    if rng.random() < 0.7:
        sigma = rng.uniform(1.0, 5.0)
        arr += int(round(rng.normalvariate(0, sigma)))
    if rng.random() < 0.35:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1.0, 4.0), arr.shape)
        arr += noise.astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def render_label(label: str, font_path: Path, rng: random.Random) -> Image.Image:
    font_size = rng.randint(24, 42)
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = rng.randint(8, 24)
    pad_y = rng.randint(4, 12)
    width = max(48, text_w + pad_x * 2)
    height = max(28, text_h + pad_y * 2)
    bg = rng.choice([(255, 255, 255), (250, 250, 246), (252, 249, 238), (246, 246, 246)])
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    text_color = rng.choice([(0, 0, 0), (18, 18, 18), (32, 32, 32), (55, 55, 55)])
    x = pad_x - bbox[0] + rng.randint(-2, 2)
    y = pad_y - bbox[1] + rng.randint(-2, 2)
    draw.text((x, y), label, font=font, fill=text_color)

    if rng.random() < 0.18:
        y_line = rng.randint(0, max(0, height - 1))
        draw.line([(0, y_line), (width, y_line)], fill=(220, 220, 220), width=1)
    if rng.random() < 0.14:
        x_line = rng.randint(0, max(0, width - 1))
        draw.line([(x_line, 0), (x_line, height)], fill=(225, 225, 225), width=1)

    if rng.random() < 0.55:
        angle = rng.uniform(-1.6, 1.6)
        image = image.rotate(angle, expand=True, fillcolor=bg)
    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.55)))
    if rng.random() < 0.75:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.32))
    if rng.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.86, 1.16))
    image = add_noise(image, rng)
    return image


def save_jpeg(image: Image.Image, path: Path, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=rng.randint(72, 96), subsampling=rng.choice([0, 1, 2]))


def build_split(
    output: Path,
    split: str,
    count: int,
    rng: random.Random,
    fonts: list[Path],
    allowed: set[str],
    max_len: int,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for index in range(count):
        label = sample_label(rng, allowed, max_len)
        image = render_label(label, rng.choice(fonts), rng)
        rel_path = Path("images") / split / f"{split}_{index:06d}.jpg"
        save_jpeg(image, output / rel_path, rng)
        rows.append((rel_path.as_posix(), label))
        if (index + 1) % 1000 == 0:
            print(f"{split}: {index + 1}/{count}")
    return rows


def write_label_file(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for rel_path, label in rows:
            file.write(f"{rel_path}\t{label}\n")


def write_preview(output: Path, rows: list[tuple[str, str]], preview_count: int) -> None:
    items = []
    for rel_path, label in rows[:preview_count]:
        src = (output / rel_path).resolve().as_posix()
        items.append(
            "<tr>"
            f"<td><img src=\"file:///{html.escape(src)}\"></td>"
            f"<td>{html.escape(label)}</td>"
            "</tr>"
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>Synthetic PaddleOCR Preview</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; }
td { border-bottom: 1px solid #ddd; padding: 8px; vertical-align: middle; }
img { max-height: 72px; max-width: 520px; image-rendering: auto; }
</style>
<h1>Synthetic PaddleOCR Preview</h1>
<table>
""" + "\n".join(items) + "\n</table>\n"
    (output / "preview.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    allowed = load_allowed_chars(args.dict_path)
    fonts = available_fonts()
    shutil.copyfile(args.dict_path, args.output / "dict.txt")

    train_rows = build_split(args.output, "train", args.train_count, rng, fonts, allowed, args.max_label_length)
    val_rows = build_split(args.output, "val", args.val_count, rng, fonts, allowed, args.max_label_length)
    write_label_file(args.output / "train.txt", train_rows)
    write_label_file(args.output / "val.txt", val_rows)
    write_preview(args.output, train_rows, args.preview_count)

    metadata = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "seed": args.seed,
        "max_label_length": args.max_label_length,
        "fonts": [str(path) for path in fonts],
        "dict_path": str(args.dict_path),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
