import csv
import json
import sys


def tsv_to_json(input_file, output_file):
    data = []

    with open(input_file, "r", encoding="utf-8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            item = {"image_id": row[0], "caption_eng": row[1], "caption_rus": ""}
            data.append(item)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_tsv_file> <output_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    tsv_to_json(input_file, output_file)
