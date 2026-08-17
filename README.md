# Data Processing GUI

Windows GUI for Neuropixels preprocessing with CatGT/TPrime, the Allen Institute `ecephys_spike_sorting` pipeline, Kilosort, stimulus metadata extraction, and preprocessing-output packaging.

## Installation

```bash
git clone --recurse-submodules https://github.com/smazonyrobak/Data_processing_GUI.git
cd Data_processing_GUI
python -m pip install -r requirements.txt
```

CatGT, TPrime, C_Waves, MATLAB/NPY-MATLAB, and the selected Kilosort implementation are external tools and must be installed separately. Configure their paths and the recording/output directories in the GUI before running a pipeline.

## Run

```bash
python preprocessing_gui.py
```

On Windows, `Data_processing_GUI_launcher.pyw` launches the same interface without a console window.

The JSON files under `pipeline configs/` are working configuration examples. Machine-generated logs, caches, shortcuts, and per-run configuration snapshots are intentionally excluded from Git.
