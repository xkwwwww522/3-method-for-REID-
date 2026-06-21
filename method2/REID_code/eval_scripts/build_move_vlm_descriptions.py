import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BASE_URL = "https://www.autodl.art/api/v1"


SYSTEM_PROMPT = """You are a careful pedestrian appearance describer for person re-identification.
Given one pedestrian image, write one short English description that focuses only on visible appearance cues.

Requirements:
- Describe only observable appearance.
- Focus on clothing color, upper-body pattern, sleeve length, lower-body type/length, shoes, bag, hat, and occlusion if visible.
- Do not guess identity, gender, age, ethnicity, mood, or action.
- Keep the sentence concise and retrieval-oriented.
- Prefer one sentence, around 12 to 28 words.
- If some details are unclear, omit them instead of guessing.
- Output only the final description text, with no prefix or explanation.
"""


USER_PROMPT = """Describe this pedestrian image for person re-identification.

Use a concise style like:
"a person wearing a black upper garment, long pants, and gray shoes"

Only output one short English description.
"""


STRUCTURED_FIELDS = {
    "upper_color": {
        "black",
        "white",
        "gray",
        "blue",
        "red",
        "green",
        "brown",
        "yellow",
        "pink",
        "purple",
        "orange",
        "beige",
        "other",
        "unknown",
    },
    "lower_color": {
        "black",
        "white",
        "gray",
        "blue",
        "red",
        "green",
        "brown",
        "yellow",
        "pink",
        "purple",
        "orange",
        "beige",
        "other",
        "unknown",
    },
    "upper_pattern": {"solid", "stripe", "plaid", "graphic", "multicolor", "unknown"},
    "sleeve_length": {"short", "long", "sleeveless", "unknown"},
    "lower_type": {"pants", "shorts", "skirt", "dress", "unknown"},
    "lower_length": {"short", "long", "unknown"},
    "shoe_color": {"black", "white", "gray", "brown", "blue", "red", "other", "unknown"},
    "backpack": {"yes", "no", "unknown"},
    "hat": {"yes", "no", "unknown"},
    "occlusion": {"none", "partial", "heavy", "unknown"},
}


STRUCTURED_SYSTEM_PROMPT = """You are a careful pedestrian attribute annotator for person re-identification.
Given one pedestrian image, return only one JSON object with normalized attribute values.

Rules:
- Use only visible appearance cues.
- Do not guess identity, gender, age, ethnicity, mood, or action.
- Do not mention background or scene context.
- If an attribute is unclear, set it to "unknown".
- Output only valid JSON and nothing else.
"""


STRUCTURED_USER_PROMPT = """Analyze this pedestrian image and output one JSON object with exactly these keys:
upper_color, lower_color, upper_pattern, sleeve_length, lower_type, lower_length, shoe_color, backpack, hat, occlusion

Allowed values:
- upper_color: black, white, gray, blue, red, green, brown, yellow, pink, purple, orange, beige, other, unknown
- lower_color: black, white, gray, blue, red, green, brown, yellow, pink, purple, orange, beige, other, unknown
- upper_pattern: solid, stripe, plaid, graphic, multicolor, unknown
- sleeve_length: short, long, sleeveless, unknown
- lower_type: pants, shorts, skirt, dress, unknown
- lower_length: short, long, unknown
- shoe_color: black, white, gray, brown, blue, red, other, unknown
- backpack: yes, no, unknown
- hat: yes, no, unknown
- occlusion: none, partial, heavy, unknown

Return only the JSON object.
"""


def build_client(args):
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Pass --api_key or set environment variable {args.api_key_env}."
        )

    return OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )


def image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    content = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{content}"


def build_messages(mode: str, image_url: str):
    if mode == "structured":
        system_prompt = STRUCTURED_SYSTEM_PROMPT
        user_prompt = STRUCTURED_USER_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    ]


def build_image_index(image_root: Optional[str]) -> Dict[str, Dict[str, object]]:
    name_index: Dict[str, Path] = {}
    stem_buckets: Dict[str, List[Path]] = {}
    search_roots = []

    if image_root:
        search_roots.append(Path(image_root))
    search_roots.append(Path("data"))

    visited = set()
    for root in search_roots:
        root = root.resolve()
        if root in visited or not root.exists():
            continue
        visited.add(root)

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            resolved = path.resolve()
            name_index.setdefault(path.name, resolved)
            stem_buckets.setdefault(path.stem, []).append(resolved)

    unique_stem_index = {}
    for stem, paths in stem_buckets.items():
        if len(paths) == 1:
            unique_stem_index[stem] = paths[0]

    return {
        "name_index": name_index,
        "stem_index": unique_stem_index,
        "stem_buckets": stem_buckets,
    }


def resolve_image_path(
    entry,
    split_name: str,
    image_root: Optional[str],
    image_index: Dict[str, Dict[str, object]],
) -> Path:
    raw_path = Path(entry["image_path"])
    if raw_path.exists():
        return raw_path

    image_name = entry.get("image_name")
    if not image_name:
        raise FileNotFoundError(f"Image path does not exist and image_name is missing: {raw_path}")

    candidate_paths = []

    if image_root:
        root = Path(image_root)
        candidate_paths.extend(
            [
                root / split_name / image_name,
                root / image_name,
            ]
        )

    if split_name == "query":
        candidate_paths.append(Path("data/move_eval_cam/query") / image_name)
    elif split_name == "gallery":
        candidate_paths.append(Path("data/move_eval_cam/gallery") / image_name)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    name_index = image_index["name_index"]
    stem_index = image_index["stem_index"]

    indexed_path = name_index.get(image_name)
    if indexed_path and indexed_path.exists():
        return indexed_path

    stem_match = stem_index.get(Path(image_name).stem)
    if stem_match and stem_match.exists():
        return stem_match

    similar = []
    stem_prefix = Path(image_name).stem[:10]
    for candidate_name in name_index.keys():
        if candidate_name.startswith(stem_prefix):
            similar.append(candidate_name)
        if len(similar) >= 5:
            break

    tried = ", ".join(str(path) for path in candidate_paths) if candidate_paths else "none"
    raise FileNotFoundError(
        f"Could not resolve image for {image_name}. "
        f"Original path: {raw_path}. Tried: {tried}. "
        f"Indexed exact match: {indexed_path}. "
        f"Indexed stem match: {stem_match}. "
        f"Similar indexed names: {similar}"
    )


def request_output(
    client,
    model: str,
    image_path: Path,
    mode: str,
    max_retries: int,
    sleep_seconds: float,
):
    image_url = image_to_data_url(image_path)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=build_messages(mode, image_url),
            )
            text = (response.output_text or "").strip()
            if not text:
                raise ValueError("Empty output returned from model.")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to generate output for {image_path}: {last_error}") from last_error


def normalize_text_description(text: str) -> str:
    return " ".join(text.strip().split())


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_structured_attributes(raw_obj: dict) -> dict:
    attrs = {}
    for key, allowed_values in STRUCTURED_FIELDS.items():
        value = str(raw_obj.get(key, "unknown")).strip().lower()
        attrs[key] = value if value in allowed_values else "unknown"
    return attrs


def has_existing_structured_attributes(entry) -> bool:
    attrs = entry.get("attributes", {})
    for key in STRUCTURED_FIELDS:
        value = str(attrs.get(key, "")).strip().lower()
        if value and value != "unknown":
            return True
    return False


def maybe_limit(entries, limit: int):
    if limit <= 0:
        return entries
    return entries[:limit]


def process_entries(
    client,
    entries,
    split_name: str,
    model: str,
    mode: str,
    image_root: Optional[str],
    image_index: Dict[str, Path],
    overwrite: bool,
    max_retries: int,
    sleep_seconds: float,
    verbose: bool,
):
    for idx, entry in enumerate(entries, start=1):
        image_path = resolve_image_path(
            entry,
            split_name=split_name,
            image_root=image_root,
            image_index=image_index,
        )
        if mode == "structured":
            exists = has_existing_structured_attributes(entry)
        else:
            exists = bool(str(entry.get("description", "")).strip())

        if exists and not overwrite:
            if verbose:
                print(f"[skip] {idx:04d} {image_path.name} -> existing output kept")
            continue

        output_text = request_output(
            client=client,
            model=model,
            image_path=image_path,
            mode=mode,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        if mode == "structured":
            raw_obj = extract_json_object(output_text)
            attrs = dict(entry.get("attributes", {}))
            attrs.update(normalize_structured_attributes(raw_obj))
            entry["attributes"] = attrs
            entry["description"] = ""
            if verbose:
                print(f"[ok]   {idx:04d} {image_path.name} -> {json.dumps(attrs, ensure_ascii=False)}")
        else:
            description = normalize_text_description(output_text)
            entry["description"] = description
            if verbose:
                print(f"[ok]   {idx:04d} {image_path.name} -> {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT-5.4 image descriptions for Move annotations and write them into the description field."
    )
    parser.add_argument("--input", required=True, type=str, help="Input annotation JSON.")
    parser.add_argument("--output", required=True, type=str, help="Output annotation JSON with VLM descriptions.")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, type=str)
    parser.add_argument("--api_key", default=None, type=str)
    parser.add_argument("--api_key_env", default="AUTODL_API_KEY", type=str)
    parser.add_argument(
        "--mode",
        default="description",
        choices=["description", "structured"],
        help="Generate free-form description text or structured attributes.",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        type=str,
        help="Optional server-side Move image root, e.g. /root/autodl-tmp/REID/data/move_eval_cam",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing description fields.")
    parser.add_argument("--query_limit", default=0, type=int, help="Only process the first N query images; 0 means all.")
    parser.add_argument("--gallery_limit", default=0, type=int, help="Only process the first N gallery images; 0 means all.")
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--sleep_seconds", default=1.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(src.read_text())
    client = build_client(args)
    image_index = build_image_index(args.image_root)
    print(
        "Indexed "
        f"{len(image_index['name_index'])} exact names and "
        f"{len(image_index['stem_index'])} unique stems for path resolution."
    )

    query_entries = maybe_limit(raw["query"], args.query_limit)
    gallery_entries = maybe_limit(raw["gallery"], args.gallery_limit)

    process_entries(
        client=client,
        entries=query_entries,
        split_name="query",
        model=args.model,
        mode=args.mode,
        image_root=args.image_root,
        image_index=image_index,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
    )
    process_entries(
        client=client,
        entries=gallery_entries,
        split_name="gallery",
        model=args.model,
        mode=args.mode,
        image_root=args.image_root,
        image_index=image_index,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
    )

    result = {
        "schema": raw.get("schema", {}),
        "query": raw["query"],
        "gallery": raw["gallery"],
    }
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"Saved VLM-described annotations to {dst}")
    print(f"query entries total: {len(raw['query'])}")
    print(f"gallery entries total: {len(raw['gallery'])}")
    if args.query_limit > 0:
        print(f"query processed this run: {len(query_entries)}")
    if args.gallery_limit > 0:
        print(f"gallery processed this run: {len(gallery_entries)}")


if __name__ == "__main__":
    main()
