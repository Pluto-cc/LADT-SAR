import argparse
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VOC-style robndbox XML annotations to YOLO OBB labels."
    )
    parser.add_argument("--xml-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-name", default="ship")
    parser.add_argument("--class-id", default=0, type=int)
    parser.add_argument(
        "--angle-unit", choices=("radians", "degrees"), default="radians"
    )
    return parser.parse_args()


def image_index(image_dir):
    index = {}
    duplicates = set()
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in index:
            duplicates.add(path.stem)
        else:
            index[path.stem] = path
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        raise RuntimeError(f"duplicate image stems in {image_dir}: {examples}")
    return index


def candidate_stems(root, xml_path):
    values = [
        root.findtext("filename", default="").strip(),
        root.findtext("path", default="").strip(),
        xml_path.stem,
    ]
    result = []
    for value in values:
        if value:
            stem = Path(value).stem
            if stem not in result:
                result.append(stem)
    return result


def rotated_corners(cx, cy, width, height, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local = (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    )
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in local
    ]


def horizontal_corners(box):
    xmin = float(box.findtext("xmin"))
    ymin = float(box.findtext("ymin"))
    xmax = float(box.findtext("xmax"))
    ymax = float(box.findtext("ymax"))
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def normalize_corners(corners, image_width, image_height):
    normalized = []
    clipped = False
    for x, y in corners:
        clipped_x = min(max(x, 0.0), float(image_width))
        clipped_y = min(max(y, 0.0), float(image_height))
        clipped = clipped or clipped_x != x or clipped_y != y
        normalized.extend((clipped_x / image_width, clipped_y / image_height))
    return normalized, clipped


def convert_xml(xml_path, image_stem, class_name, class_id, angle_unit):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", default="0")))
    height = int(float(root.findtext("size/height", default="0")))
    if width <= 0 or height <= 0:
        raise ValueError("missing or invalid image size")

    lines = []
    clipped_count = 0
    for obj in root.findall("object"):
        if obj.findtext("name", default="").strip().lower() != class_name.lower():
            continue

        rotated = obj.find("robndbox")
        horizontal = obj.find("bndbox")
        if rotated is not None:
            cx = float(rotated.findtext("cx"))
            cy = float(rotated.findtext("cy"))
            box_width = float(rotated.findtext("w"))
            box_height = float(rotated.findtext("h"))
            angle = float(rotated.findtext("angle"))
            if angle_unit == "degrees":
                angle = math.radians(angle)
            if box_width <= 0 or box_height <= 0:
                raise ValueError("non-positive rotated-box width or height")
            corners = rotated_corners(cx, cy, box_width, box_height, angle)
        elif horizontal is not None:
            corners = horizontal_corners(horizontal)
        else:
            raise ValueError(f"{class_name} object has neither robndbox nor bndbox")

        values, clipped = normalize_corners(corners, width, height)
        clipped_count += int(clipped)
        coordinates = " ".join(f"{value:.8f}" for value in values)
        lines.append(f"{class_id} {coordinates}")

    return lines, clipped_count


def main():
    args = parse_args()
    if not args.xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory does not exist: {args.xml_dir}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {args.image_dir}")

    images = image_index(args.image_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter(images=len(images))
    matched_images = set()

    xml_paths = sorted(args.xml_dir.rglob("*.xml"))
    stats["xml_files"] = len(xml_paths)
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        stems = candidate_stems(root, xml_path)
        image_stem = next((stem for stem in stems if stem in images), None)
        if image_stem is None:
            stats["xml_without_split_image"] += 1
            continue
        if image_stem in matched_images:
            raise RuntimeError(f"multiple XML files match image: {images[image_stem]}")

        try:
            lines, clipped = convert_xml(
                xml_path,
                image_stem,
                args.class_name,
                args.class_id,
                args.angle_unit,
            )
        except (ET.ParseError, TypeError, ValueError) as exc:
            stats["invalid_xml"] += 1
            print(f"Warning: skipped {xml_path}: {exc}")
            continue

        label_path = args.output_dir / f"{image_stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        matched_images.add(image_stem)
        stats["matched_images"] += 1
        stats["ship_boxes"] += len(lines)
        stats["empty_ship_labels"] += int(not lines)
        stats["clipped_boxes"] += clipped

    stats["images_without_xml"] = len(set(images) - matched_images)
    print("Conversion complete")
    for key in (
        "images",
        "xml_files",
        "matched_images",
        "ship_boxes",
        "empty_ship_labels",
        "clipped_boxes",
        "images_without_xml",
        "xml_without_split_image",
        "invalid_xml",
    ):
        print(f"{key}: {stats[key]}")

    if stats["images_without_xml"] or stats["invalid_xml"]:
        raise SystemExit(
            "Conversion finished with unmatched or invalid annotations; inspect the counts above."
        )


if __name__ == "__main__":
    main()
