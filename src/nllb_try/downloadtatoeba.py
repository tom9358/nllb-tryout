import os
import requests
import shutil
import bz2
import tarfile
from tqdm.auto import tqdm


def _ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)

def download_and_unpack_bz2(link: str, output_file: str):
    response = requests.get(link, stream=True)
    compressed_file = output_file + ".bz2"

    with open(compressed_file, "wb") as file_out:
        shutil.copyfileobj(response.raw, file_out)

    with bz2.BZ2File(compressed_file, 'rb') as file_in, open(output_file, 'wb') as uncompressed_out:
        shutil.copyfileobj(file_in, uncompressed_out)
    print(f'Unpacked to {output_file}.')

def download_and_unpack_links_file(tatoeba_path: str, redownload: bool = False):
    links_file_url = r'https://downloads.tatoeba.org/exports/links.tar.bz2'
    tar_file = os.path.join(tatoeba_path, "links.tar")
    csv_file = os.path.join(tatoeba_path, "links.csv")

    if redownload or not os.path.exists(csv_file):
        _ensure_directory(tatoeba_path)
        download_and_unpack_bz2(links_file_url, tar_file)

        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(tatoeba_path)
        print(f'Links CSV unpacked to {tatoeba_path}.')
    else:
        print('Links CSV already exists. Skipping download and unpacking.')

def download_and_unpack_tatoeba(tatoeba_path: str, source_lang: str, redownload: bool = False):
    sentence_file_url = rf'https://downloads.tatoeba.org/exports/per_language/{source_lang}/{source_lang}_sentences.tsv.bz2'
    unpacked_file = os.path.join(tatoeba_path, f"{source_lang}_sentences.tsv")

    if redownload or not os.path.exists(unpacked_file):
        _ensure_directory(tatoeba_path)
        download_and_unpack_bz2(sentence_file_url, unpacked_file)
    else:
        print(f'{unpacked_file} already exists. Skipping download and unpacking.')

def main_download(source_langs, redownload: bool = False, tatoeba_path: str = "data/tatoeba"):
    print('Downloading necessary Tatoeba files...')
    download_and_unpack_links_file(tatoeba_path=tatoeba_path, redownload=redownload)
    for src_lang in tqdm(source_langs):
        download_and_unpack_tatoeba(tatoeba_path=tatoeba_path, source_lang=src_lang, redownload=redownload)