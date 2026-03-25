import os
import subprocess
import bz2
from pathlib import Path

# Directories
DUMP_DIR = Path("data/wiki_dump")
TEXT_DIR = Path("data/wiki_text")
CORPUS_FILE = Path("data/corpus.txt")

DUMP_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Wikipedia dump URLs and paths
WIKI_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
DUMP_BZ2_PATH = DUMP_DIR / "enwiki-latest-pages-articles.xml.bz2"
DUMP_XML_PATH = DUMP_DIR / "enwiki-latest-pages-articles.xml"

def download_wikipedia():
    if DUMP_BZ2_PATH.exists():
        print(f"Dump already exists: {DUMP_BZ2_PATH}")
        return

    cmd = ["wget", "-O", str(DUMP_BZ2_PATH), WIKI_URL]
    print("Downloading Wikipedia dump...")
    subprocess.run(cmd, check=True)

def decompress_bz2():
    if DUMP_XML_PATH.exists():
        print(f"XML dump already exists: {DUMP_XML_PATH}")
        return

    print(f"Decompressing {DUMP_BZ2_PATH} to {DUMP_XML_PATH}...")
    with bz2.open(DUMP_BZ2_PATH, "rb") as f_in, open(DUMP_XML_PATH, "wb") as f_out:
        for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
            f_out.write(chunk)
    print("Decompression complete.")

def extract_text():
    print("Extracting text with WikiExtractor...")
    cmd = [
        "wikiextractor",
        "--json",
        "--no-templates",
        "-o", str(TEXT_DIR),
        str(DUMP_XML_PATH)
    ]
    subprocess.run(cmd, check=True)

def combine_text():
    print("Combining into single corpus.txt...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(TEXT_DIR):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        for line in f:
                            out.write(line + "\n")
    print(f"Saved corpus at {CORPUS_FILE}")

if __name__ == "__main__":
    download_wikipedia()
    decompress_bz2()
    extract_text()
    combine_text()
