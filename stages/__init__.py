__all__ = [
    "run_ecephys_spike_sorting",
    "run_catgt_only",
    "run_custom_ks4",
    "run_state_sorter",
    "build_stimulus_metadata",
    "build_preprocessing_output",
]


def __getattr__(name):
    if name == "run_ecephys_spike_sorting":
        from .ecephys_pipeline import run_ecephys_spike_sorting

        return run_ecephys_spike_sorting
    if name == "run_catgt_only":
        from .ecephys_pipeline import run_catgt_only

        return run_catgt_only
    if name == "run_custom_ks4":
        from .custom_ks4 import run_custom_ks4

        return run_custom_ks4
    if name == "run_state_sorter":
        from .state_sorter import run_state_sorter

        return run_state_sorter
    if name == "build_stimulus_metadata":
        from .stimulus_metadata import build_stimulus_metadata

        return build_stimulus_metadata
    if name == "build_preprocessing_output":
        from .build_preprocessing_output import build_preprocessing_output

        return build_preprocessing_output
    raise AttributeError(name)
