import os
import requests
import bz2
import shutil
from tqdm.auto import tqdm

def download_and_unpack_bz2(link: str, output_file: str):
    """
    Download a bz2 file from the given link and unpack it.
    """
    # Download the bz2 compressed file
    response = requests.get(link, stream=True)
    compressed_file = output_file + ".bz2"
    
    with open(compressed_file, "wb") as file_out:
        shutil.copyfileobj(response.raw, file_out)
    
    # Unpack the bz2 file
    with bz2.BZ2File(compressed_file, 'rb') as file_in, open(output_file, 'wb') as uncompressed_out:
        shutil.copyfileobj(file_in, uncompressed_out)

def download_tatoeba(source_lang: str, trt_lang: str, redownload: bool = False):
    links = [
        f'https://downloads.tatoeba.org/exports/per_language/{source_lang}/{source_lang}_sentences.tsv.bz2',
        f'https://downloads.tatoeba.org/exports/per_language/{trt_lang}/{trt_lang}_sentences.tsv.bz2',
        'https://downloads.tatoeba.org/exports/links.tar.bz2'
    ]
    files = [
        source_lang + "_sentences.tsv",
        trt_lang + "_sentences.tsv",
        "links.tar"
    ]
    
    for link, file in zip(links, files):
        if not os.path.exists(file) or redownload:
            print(f'Downloading and unpacking {file}...')
            download_and_unpack_bz2(link, file)
        else:
            print(f'File {file} already exists. Skipping download.')

def main_download(source_langs):
    print('Downloading necessary Tatoeba files...')
    for src_lang in tqdm(source_langs):
        download_tatoeba(src_lang, src_lang)