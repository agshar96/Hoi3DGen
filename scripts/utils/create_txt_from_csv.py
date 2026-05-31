import argparse
import csv

SUFFIX = " White Background, Full Body Visible."

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create prompts.txt from the Description column of a CSV file."
    )
    parser.add_argument(
        "--input_csv",
        help="Path to the input CSV file, e.g. prompts_deconstructed.csv"
    )
    parser.add_argument(
        "--output_txt",
        help="Path to the output TXT file, e.g. prompts.txt"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input_csv, mode="r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if "Description" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'Description' column.")

        prompts = []

        for row in reader:
            description = row.get("Description", "").strip()

            if description:
                prompts.append(description + SUFFIX)

    with open(args.output_txt, mode="w", encoding="utf-8") as txt_file:
        for prompt in prompts:
            txt_file.write(prompt + "\n")

    print(f"Saved {len(prompts)} prompts to {args.output_txt}")

if __name__ == "__main__":
    main()
