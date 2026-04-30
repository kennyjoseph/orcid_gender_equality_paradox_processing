#!/usr/bin/env python3
"""Flag cmd values from out.csv that are non-English or not academic fields/departments."""

import csv
import json
import re
import sys

import anthropic

BATCH_SIZE = 200
INPUT_FILE = "top_250_per_country.csv"
OUTPUT_FILE = "flagged_for_potential_removal.csv"

SYSTEM_PROMPT = (
    "You are a classifier for academic department/field names. "
    "You will receive a numbered list of strings. "
    "For each string, determine if it meets either condition:\n"
    "  1. NOT written in English (e.g. French, Spanish, Arabic, Chinese, etc.)\n"
    "  2. Clearly NOT an academic field or university department "
    "(e.g. a person's name, a city or country name, a bare number, a company name, "
    "random text, a non-academic abbreviation, etc.)\n\n"
    "Return ONLY a JSON array. Each element must have:\n"
    '  "index": 1-based position in the list\n'
    '  "value": the original string\n'
    '  "reason": "non_english", "not_academic", or "both"\n\n'
    "Include only strings that fail at least one check. "
    "If all strings pass, return an empty array []. "
    "Output no text outside the JSON array."
)


def get_unique_cmds(filepath: str) -> list[str]:
    cmds: set[str] = set()
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row["cmd"].strip()
            if val:
                cmds.add(val)
    return sorted(cmds)


def flag_batch(client: anthropic.Anthropic, batch: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Classify these strings:\n\n{numbered}"}],
    )

    text = next(b.text for b in response.content if b.type == "text")
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
    cmds = get_unique_cmds(INPUT_FILE)
    print(f"Found {len(cmds)} unique cmd values.")

    all_flagged: list[dict] = []
    total_batches = (len(cmds) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(cmds), BATCH_SIZE):
        batch = cmds[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} items)...", end=" ", flush=True)

        flagged = flag_batch(client, batch)
        for item in flagged:
            all_flagged.append({"cmd": item["value"], "reason": item["reason"]})
        print(f"{len(flagged)} flagged.")

    all_flagged.sort(key=lambda x: x["cmd"])

    print(f"\nWriting {len(all_flagged)} flagged entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cmd", "reason"])
        writer.writeheader()
        writer.writerows(all_flagged)

    print("Done.")


if __name__ == "__main__":
    main()
