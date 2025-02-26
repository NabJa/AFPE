import argparse
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import List
from zipfile import ZipFile


def determine_file_type(file_path: Path) -> str:
    """
    Determine the file type of the file at the file path
    """
    return "".join(file_path.suffixes)


def unzip_file(file_path: Path) -> None:
    """
    Unzip the file at the file path to the output path
    """
    output_path = file_path.parent / file_path.stem

    with ZipFile(file_path, "r") as zip_obj:
        zip_obj.extractall(output_path)


def untar_file(file_path: Path) -> None:
    """
    Unzip the file at the file path to the output path
    """
    output_path = file_path.parent / file_path.name.split(".")[0]

    with tarfile.open(file_path, "r") as tar_obj:
        tar_obj.extractall(output_path)


def extract_files(file_path: Path) -> None:
    """
    Extract the files at the file path to the output path
    """
    file_type = determine_file_type(file_path)

    if file_type == ".zip":
        unzip_file(file_path)
    elif file_type in [".tar", ".tar.gz"]:
        untar_file(file_path)
    else:
        raise ValueError("File type not supported")


def get_files_to_extract(data_path: Path) -> List[Path]:
    """Extract all files in the data path. If the data path is a file, return that file."""
    if data_path.is_file():
        return [data_path]

    file_paths = []
    for suffixes in [[".zip"], [".tar"], [".tar.gz"]]:
        file_paths += list(data_path.glob("*" + suffixes[0]))
    return file_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--num_processes", type=int, default=12)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    files = get_files_to_extract(args.data_path)

    with Pool(args.num_processes) as p:
        p.map(extract_files, files)
