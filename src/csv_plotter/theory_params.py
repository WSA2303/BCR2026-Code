# src/csv_plotter/theory_params.py

THEORY_PRESETS = {
    "0.3": {
        "ty": 39.81,
        "Kn": 0.91,
        "n": 1.0,
        "rho": 1560.0,
        "th_deg": 15.0,
        "h": 0.033482,
        "dz": None,
        "adm": False,
    },
    "0.9": {
        "ty": 119.42,
        "Kn": 0.91,
        "n": 1.0,
        "rho": 1560.0,
        "th_deg": 15.0,
        "h": 0.033482,
        "dz": None,
        "adm": False,
    },
}

DEFAULT_C = 0.9

# Mantém THEORY_PARAMS existindo para não quebrar imports antigos
THEORY_PARAMS = THEORY_PRESETS[f"{DEFAULT_C:.1f}"].copy()


def get_theory_params(C: float) -> dict:
    key = f"{C:.1f}"  # evita problemas com float
    if key not in THEORY_PRESETS:
        raise ValueError(f"C inválido: {C}. Use um destes: {list(THEORY_PRESETS.keys())}")
    return THEORY_PRESETS[key].copy()


def set_theory_params(C: float) -> None:
    """Atualiza THEORY_PARAMS em-place (sem quebrar quem importou THEORY_PARAMS)."""
    new_params = get_theory_params(C)
    THEORY_PARAMS.clear()
    THEORY_PARAMS.update(new_params)