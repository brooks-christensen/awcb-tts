from pathlib import Path
import re
import argparse


def extract_section_lines(md_text: str, section_letter: str) -> list[str]:
    lines = md_text.splitlines()

    start_pat = re.compile(
        rf"^##\s+Section\s+{re.escape(section_letter)}\b",
        flags=re.IGNORECASE,
    )
    next_section_pat = re.compile(r"^##\s+Section\s+[A-Z]\b")

    in_section = False
    extracted = []

    for raw_line in lines:
        line = raw_line.strip()

        if not in_section:
            if start_pat.match(line):
                in_section = True
            continue

        # stop when the next section starts
        if next_section_pat.match(line):
            break

        # keep only numbered utterance lines like:
        # 1. A quiet mind can notice the smallest change in tone.
        m = re.match(r"^\d+\.\s+(.*\S)\s*$", line)
        if m:
            extracted.append(m.group(1).strip())

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to lecture_style_tts_recording_script.md",
    )
    parser.add_argument(
        "--section",
        default="A",
        help="Section letter to extract, e.g. A, B, C, or D",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output text file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    md_text = input_path.read_text(encoding="utf-8")
    lines = extract_section_lines(md_text, args.section.upper())

    if not lines:
        raise ValueError(f"No numbered lines found for Section {args.section}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    main()