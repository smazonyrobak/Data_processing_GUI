from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
BASE_URL = "https://github.com/smazonyrobak/Data_processing_GUI/releases/download/runtime-assets-v1"
ASSETS = {
    "neuropixels-processing-tools-win-v1.zip": "e426f823a0f7e8105e68d31844c781994e1ff415b5f81526d72d75c43e5a8049",
    "ecephys_spike_sorting_LNE-v1.zip": "8966f8bbf47b62dc766c3f2eb1fefbef12c7afaeff5886727d452ff0516ee431",
}
DOWNLOADS = ROOT / ".runtime-downloads"


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


DOWNLOADS.mkdir(exist_ok=True)
for filename, expected in ASSETS.items():
    archive = DOWNLOADS / filename
    if not archive.is_file() or digest(archive) != expected:
        temporary = archive.with_suffix(".part")
        print(f"downloading: {filename}")
        request = Request(f"{BASE_URL}/{filename}", headers={"User-Agent": "data-processing-gui-setup"})
        with urlopen(request) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if digest(temporary) != expected:
            temporary.unlink()
            raise RuntimeError(f"SHA-256 mismatch for {filename}")
        temporary.replace(archive)
    destination = ROOT / "tools" if filename.startswith("neuropixels-") else ROOT
    print(f"extracting: {filename}")
    with ZipFile(archive) as bundle:
        bundle.extractall(destination)

required = [
    ROOT / "tools/CatGT-win/CatGT.exe",
    ROOT / "tools/TPrime-win/TPrime.exe",
    ROOT / "tools/C_Waves-win/C_Waves.exe",
    ROOT / "ecephys_spike_sorting_LNE/ecephys_spike_sorting",
]
if not all(path.exists() for path in required):
    raise RuntimeError("Runtime extraction is incomplete")

print("Neuropixels processing tools and the LNE ecephys pipeline are installed.")
