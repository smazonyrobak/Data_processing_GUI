# Data Processing GUI

Windows GUI for Neuropixels preprocessing with CatGT/TPrime, the Allen Institute `ecephys_spike_sorting` pipeline, Kilosort, stimulus metadata extraction, and preprocessing-output packaging.

## Installation

```bash
git clone https://github.com/smazonyrobak/Data_processing_GUI.git
cd Data_processing_GUI
python setup_runtime.py
python -m pip install -r requirements.txt
```

`setup_runtime.py` downloads and checksum-verifies the same Windows CatGT, TPrime, and C_Waves distributions used by the working installation, plus the synchronized LNE ecephys pipeline. Their paths are selected automatically for new configurations. MATLAB/NPY-MATLAB and legacy MATLAB Kilosort variants remain optional external tools; the Python Kilosort 4 path is installed from `requirements.txt`.

## Run

```bash
python preprocessing_gui.py
```

On Windows, `Data_processing_GUI_launcher.pyw` launches the same interface without a console window.

The JSON files under `pipeline configs/` are working configuration examples. Select the recording and output folders for the current machine before running. Machine-generated logs, caches, shortcuts, environments, downloaded runtime tools, and per-run configuration snapshots are intentionally excluded from Git.
