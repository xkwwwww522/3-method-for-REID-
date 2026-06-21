import argparse
import json
import re
from pathlib import Path


MOVE_PATTERN = re.compile(r"([-\d]+)C(\d+)T(\d+)F(\d+)")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_name(img_name):
    match = MOVE_PATTERN.fullmatch(Path(img_name).stem)
    if not match:
        raise ValueError(f"Unexpected Move filename format: {img_name}")
    pid, camid, trackid, frameid = match.groups()
    return int(pid), int(camid), int(trackid), int(frameid)


def list_images(dir_path):
    images = []
    for img_path in sorted(Path(dir_path).rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_SUFFIXES:
            continue
        pid, camid, trackid, frameid = parse_name(img_path.name)
        images.append(
            {
                "image_path": str(img_path),
                "image_name": img_path.name,
                "pid": pid,
                "camid": camid,
                "trackid": trackid,
                "frameid": frameid,
                "attributes": {
                    "upper_color": "",
                    "lower_color": "",
                    "backpack": "unknown",
                    "hat": "unknown",
                    "occlusion": "unknown",
                },
                "description": "",
            }
        )
    return images


def main():
    parser = argparse.ArgumentParser(description="Build a manual annotation template for Move attribute descriptions")
    parser.add_argument("--query_dir", required=True, type=str)
    parser.add_argument("--gallery_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema": {
            "upper_color": "string, e.g. black/white/blue/red/gray",
            "lower_color": "string, e.g. black/blue/gray",
            "backpack": "yes/no/unknown",
            "hat": "yes/no/unknown",
            "occlusion": "none/partial/heavy/unknown",
            "description": "optional free-form sentence; leave empty to auto-generate from attributes",
        },
        "query": list_images(args.query_dir),
        "gallery": list_images(args.gallery_dir),
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved attribute template to {output_path}")
    print(f"query samples: {len(data['query'])}")
    print(f"gallery samples: {len(data['gallery'])}")


if __name__ == "__main__":
    main()
