import os
import csv
import json
from tqdm.auto import tqdm
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import List, Dict


def csv_to_json(source_dir: str, output_file: str) -> None:
    """
    Convert CSV files in the source directory to a single JSON file.

    :param source_dir: Directory containing CSV files.
    :param output_file: Path to the output JSON file.
    """
    records: List[Dict[str, str]] = []
    csv_files = [f for f in os.listdir(source_dir) if f.endswith(".csv")]

    for filename in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(source_dir, filename)
        try:
            with open(file_path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    image_id = row["image_name"].replace(".jpg", "")
                    record = {
                        "image_id": image_id,
                        "caption_rus": "None",
                        "caption_eng": row["caption"],
                    }
                    records.append(record)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    with open(output_file, mode="w", encoding="utf-8") as jsonfile:
        json.dump(records, jsonfile, ensure_ascii=False, indent=4)


def tsv_to_json(input_file: str, output_file: str) -> None:
    """
    Convert a TSV file to a JSON file.

    :param input_file: Path to the input TSV file.
    :param output_file: Path to the output JSON file.
    """
    data: List[Dict[str, str]] = []

    with open(input_file, "r", encoding="utf-8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            item = {"image_id": row[0], "caption_eng": row[1], "caption_rus": ""}
            data.append(item)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def tar_to_folder(source_dir: str, dest_dir: str, max_workers: int = 16) -> None:
    """
    Extract all tar files from the source directory to the destination directory.

    :param source_dir: Directory containing tar files.
    :param dest_dir: Target directory for extracted files.
    :param max_workers: Maximum number of threads for parallel extraction.
    """

    def extract_tar_file(file_path: str, dest_dir: str) -> str:
        try:
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=dest_dir)
            return file_path
        except Exception as e:
            print(f"Failed to extract {os.path.basename(file_path)}: {e}")
            return ""

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    tar_files = [
        os.path.join(source_dir, filename)
        for filename in os.listdir(source_dir)
        if filename.endswith((".tar", ".tar.gz", ".tgz"))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_tar_file, file_path, dest_dir): file_path
            for file_path in tar_files
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting tar files"
        ):
            result = future.result()
            if result:
                print(f"Extracted {os.path.basename(result)} to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Utility for processing files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_csv_to_json = subparsers.add_parser(
        "csv_to_json", help="Convert CSV files to JSON."
    )
    parser_csv_to_json.add_argument(
        "source_dir", type=str, help="Directory containing CSV files."
    )
    parser_csv_to_json.add_argument(
        "output_file", type=str, help="Path to the output JSON file."
    )

    parser_tsv_to_json = subparsers.add_parser(
        "tsv_to_json", help="Convert a TSV file to JSON."
    )
    parser_tsv_to_json.add_argument(
        "input_file", type=str, help="Path to the input TSV file."
    )
    parser_tsv_to_json.add_argument(
        "output_file", type=str, help="Path to the output JSON file."
    )

    parser_tar_to_folder = subparsers.add_parser(
        "tar_to_folder", help="Extract tar files to a folder."
    )
    parser_tar_to_folder.add_argument(
        "source_dir", type=str, help="Directory containing tar files."
    )
    parser_tar_to_folder.add_argument(
        "dest_dir", type=str, help="Target directory for extracted files."
    )
    parser_tar_to_folder.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of threads for parallel extraction.",
    )

    args = parser.parse_args()

    if args.command == "csv_to_json":
        csv_to_json(args.source_dir, args.output_file)
    elif args.command == "tsv_to_json":
        tsv_to_json(args.input_file, args.output_file)
    elif args.command == "tar_to_folder":
        tar_to_folder(args.source_dir, args.dest_dir, args.max_workers)


if __name__ == "__main__":
    main()
