import os
import requests
import shutil
import bz2
import tarfile
from tqdm.auto import tqdm

TATOEBA_PATH = 'tatoeba'

def _ensure_directory(path: str):
    """Ensure the existence of a directory."""
    os.makedirs(path, exist_ok=True)

def download_and_unpack_bz2(link: str, output_file: str):
    """
    Download a bz2 file from the given link and unpack it.
    """
    response = requests.get(link, stream=True)
    compressed_file = output_file + ".bz2"

    with open(compressed_file, "wb") as file_out:
        shutil.copyfileobj(response.raw, file_out)

    with bz2.BZ2File(compressed_file, 'rb') as file_in, open(output_file, 'wb') as uncompressed_out:
        shutil.copyfileobj(file_in, uncompressed_out)
    print(f'Unpacked to {output_file}.')

def download_and_unpack_links_file(redownload: bool = False):
    """Download and unpack the links csv if it doesn't exist or if redownload is True."""
    links_file_url = r'https://downloads.tatoeba.org/exports/links.tar.bz2'
    tar_file = os.path.join(TATOEBA_PATH, "links.tar")
    csv_file = os.path.join(TATOEBA_PATH, "links.csv")

    if redownload or not os.path.exists(csv_file):
        _ensure_directory(TATOEBA_PATH)
        download_and_unpack_bz2(links_file_url, tar_file)

        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(TATOEBA_PATH)
        print(f'Links CSV unpacked to {TATOEBA_PATH}.')
    else:
        print(f'Links CSV already exists. Skipping download and unpacking.')

def download_and_unpack_tatoeba(source_lang: str, redownload: bool = False):
    """Download and unpack source language sentences."""
    sentence_file_url = rf'https://downloads.tatoeba.org/exports/per_language/{source_lang}/{source_lang}_sentences.tsv.bz2'
    unpacked_file = os.path.join(TATOEBA_PATH, f"{source_lang}_sentences.tsv")

    if redownload or not os.path.exists(unpacked_file):
        _ensure_directory(TATOEBA_PATH)
        download_and_unpack_bz2(sentence_file_url, unpacked_file)
    else:
        print(f'{unpacked_file} already exists. Skipping download and unpacking.')

def main_download(source_langs, redownload=False):
    print('Downloading necessary Tatoeba files...')
    download_and_unpack_links_file(redownload)
    for src_lang in tqdm(source_langs):
        download_and_unpack_tatoeba(src_lang, redownload)