#!/usr/bin/env python3
"""Classify non-academic cmd values as location, acronym, organization, or other."""

import csv
import json
import re
import sys

import anthropic

BATCH_SIZE = 100
INPUT_FILE = "retained_deeper_classification_nonacademic.csv"
OUTPUT_FILE = "classified_nonacademic.csv"

SYSTEM_PROMPT = (
    "You are a classifier for strings that have already been identified as non-academic "
    "department/field names. Classify each string into exactly one of these categories:\n\n"
    "  - location: a place name — city, region, country, geographic area "
    "(e.g. 'managua', 'leon', 'chontales')\n"
    "  - acronym: an abbreviation or acronym that is not a recognizable academic term "
    "(e.g. 'raccn', 'raccs', 'cur')\n"
    "  - organization: a non-academic organization, company, institute, or proper entity name "
    "(e.g. 'red cross', 'world bank', 'microsoft')\n"
    "  - other: anything that does not clearly fit the above "
    "(e.g. common words, random text, person names, single letters)\n\n"
    "Return ONLY a JSON array. Each element must have:\n"
    '  "index": 1-based position in the list\n'
    '  "value": the original string\n'
    '  "category": one of "location", "acronym", "organization", "other"\n\n'
    "Every string must appear in the output. Output no text outside the JSON array."
)


def get_values(filepath: str) -> list[str]:
    values: list[str] = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if row:
                val = row[0].strip()
                if val:
                    values.append(val)
    return values


def classify_batch(client: anthropic.Anthropic, batch: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Classify these strings:\n\n{numbered}"}],
    )

    text = next(b.text for b in response.content if b.type == "text")
    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"  WARNING: could not parse JSON from response:\n{text[:300]}", file=sys.stderr)
        return []


def main() -> None:
    client = anthropic.Anthropic()

    print(f"Reading {INPUT_FILE}...")
    values = get_values(INPUT_FILE)
    print(f"Found {len(values)} values.")

    results: list[dict] = []
    total_batches = (len(values) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(values), BATCH_SIZE):
        batch = values[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} items)...", end=" ", flush=True)

        classified = classify_batch(client, batch)

        # index into original batch to fill any gaps the model skips
        classified_by_index = {item["index"]: item for item in classified}
        for j, val in enumerate(batch, start=1):
            item = classified_by_index.get(j, {"value": val, "category": "other"})
            results.append({"cmd": item["value"], "category": item["category"]})

        counts = {}
        for item in classified:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        print(", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))

    print(f"\nWriting {len(results)} rows to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cmd", "category"])
        writer.writeheader()
        writer.writerows(results)

    # summary
    totals: dict[str, int] = {}
    for row in results:
        totals[row["category"]] = totals.get(row["category"], 0) + 1
    print("\nSummary:")
    for cat, n in sorted(totals.items()):
        print(f"  {cat}: {n}")
    print("Done.")


if __name__ == "__main__":
    main()
