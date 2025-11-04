"""Spectral analysis pipeline for producing QC reports and Prospector-ready inputs.

This script loads a spectrum CSV, optionally corrects Milky Way extinction,
produces several diagnostic plots, and saves a summary README documenting the
run.  It also provides an optional quick Prospector SSP fit for rapid
inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global seed for reproducibility (e.g., Prospector's stochastic optimizers).
np.random.seed(42)


@dataclass
class SpectrumData:
    """Container for wavelength and flux arrays."""

    wavelength_um: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class DereddeningResult:
    """Result of the extinction correction."""

    flux: np.ndarray
    error: np.ndarray
    factor: np.ndarray
    ebv: Optional[float]
    law: Optional[str]
    rv: Optional[float]
    notes: List[str]


class ReadmeLogger:
    """Helper that accumulates lines destined for README.txt."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    def add(self, line: str) -> None:
        logging.debug("README append: %s", line)
        self._lines.append(line)

    def extend(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.add(line)

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spectral analysis report.")
    parser.add_argument("--csv", default="/mnt/data/1223468_central_spectrum.csv", help="Input spectrum CSV path")
    parser.add_argument(
        "--ext",
        default="/mnt/data/extinction_correction.py",
        help="Path to extinction correction module",
    )
    parser.add_argument("--ra", type=float, default=190.509395, help="Right Ascension in degrees")
    parser.add_argument("--dec", type=float, default=11.646909, help="Declination in degrees")
    parser.add_argument(
        "--deredden",
        choices=["true", "false"],
        default="true",
        help="Toggle Milky Way dereddening",
    )
    parser.add_argument(
        "--law",
        choices=["ccm89", "calzetti"],
        default="ccm89",
        help="Extinction law identifier",
    )
    parser.add_argument("--rv", type=float, default=3.1, help="Total-to-selective extinction R_V")
    parser.add_argument("--outdir", default="./out", help="Output directory")
    parser.add_argument(
        "--quickfit",
        choices=["true", "false"],
        default="false",
        help="Enable quick Prospector SSP fit",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_spectrum(csv_path: Path, readme: ReadmeLogger) -> SpectrumData:
    logging.info("Loading spectrum: %s", csv_path)
    if not csv_path.exists():
        msg = f"Input CSV not found: {csv_path}"
        logging.error(msg)
        readme.add(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    readme.add(f"Loaded CSV with shape {df.shape} from {csv_path}")

    if df.shape[1] < 6:
        msg = "CSV does not contain the required columns (>=6)."
        logging.error(msg)
        readme.add(msg)
        raise ValueError(msg)

    rename_map = {
        df.columns[0]: "wavelength_um",
        df.columns[4]: "F_lambda",
        df.columns[5]: "eF_lambda",
    }
    df = df.rename(columns=rename_map)

    essential_cols = ["wavelength_um", "F_lambda", "eF_lambda"]
    missing = [col for col in essential_cols if col not in df.columns]
    if missing:
        msg = f"Missing required columns after renaming: {missing}"
        logging.error(msg)
        readme.add(msg)
        raise ValueError(msg)

    before_count = len(df)
    mask_finite = np.isfinite(df[essential_cols]).all(axis=1)
    mask_error = df["eF_lambda"] > 0
    mask = mask_finite & mask_error
    cleaned_df = df.loc[mask, essential_cols]
    after_count = len(cleaned_df)
    removed = before_count - after_count

    logging.info("Valid rows: %d (removed %d)", after_count, removed)
    readme.add(f"Initial rows: {before_count}; Removed during cleaning: {removed}; Remaining: {after_count}")

    if after_count == 0:
        msg = "No valid data rows remain after filtering."
        logging.error(msg)
        readme.add(msg)
        raise ValueError(msg)

    wavelengths = cleaned_df["wavelength_um"].to_numpy(dtype=float)
    flux = cleaned_df["F_lambda"].to_numpy(dtype=float)
    error = cleaned_df["eF_lambda"].to_numpy(dtype=float)

    meta = {
        "wavelength_min_um": float(np.min(wavelengths)),
        "wavelength_max_um": float(np.max(wavelengths)),
        "flux_mean": float(np.mean(flux)),
        "flux_median": float(np.median(flux)),
        "flux_std": float(np.std(flux)),
    }
    logging.info(
        "Wavelength range: %.5f-%.5f μm; Flux mean/median/std: %.3e/%.3e/%.3e",
        meta["wavelength_min_um"],
        meta["wavelength_max_um"],
        meta["flux_mean"],
        meta["flux_median"],
        meta["flux_std"],
    )
    readme.add(
        "Wavelength range (μm): {min:.5f} - {max:.5f}; F_lambda stats: mean={mean:.3e}, median={median:.3e}, std={std:.3e}".format(
            min=meta["wavelength_min_um"],
            max=meta["wavelength_max_um"],
            mean=meta["flux_mean"],
            median=meta["flux_median"],
            std=meta["flux_std"],
        )
    )

    return SpectrumData(wavelength_um=wavelengths, flux=flux, error=error, metadata=meta)


def save_head_summary(df: pd.DataFrame, out_path: Path) -> None:
    logging.info("Saving head summary to %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_head = df.head().copy()
    df_head.to_csv(out_path, index=False)


def plot_raw_spectrum(data: SpectrumData, out_path: Path) -> None:
    logging.info("Creating raw spectrum plot: %s", out_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.errorbar(
        data.wavelength_um,
        np.abs(data.flux),
        yerr=data.error,
        fmt="o",
        markersize=3,
        lw=0.7,
        alpha=0.6,
        label="Observed |F_lambda|",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"|F$_{\lambda}$| [cgs]")
    ax.set_title("Raw Spectrum (log-log)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def import_extinction_module(module_path: Path, readme: ReadmeLogger) -> Optional[Any]:
    if not module_path.exists():
        msg = f"Extinction module not found at {module_path}"
        logging.warning(msg)
        readme.add(msg)
        return None
    try:
        spec = importlib.util.spec_from_file_location("extinction_helper", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        logging.info("Imported extinction module from %s", module_path)
        readme.add(f"Extinction module imported: {module_path}")
        return module
    except Exception as exc:
        msg = f"Failed to import extinction module: {exc}"
        logging.warning(msg)
        readme.add(msg)
        return None


def find_extinction_function(module: Any) -> Optional[Callable[..., Any]]:
    candidates = [
        "apply_mw_extinction",
        "apply_extinction_correction",
        "correct_extinction",
        "deredden_flux",
        "extinction_correction",
        "apply_mw_dereddening",
        "get_dereddening_factor",
    ]
    for name in candidates:
        func = getattr(module, name, None)
        if callable(func):
            logging.info("Using extinction correction function: %s", name)
            return func
    return None


def compute_ccm89_correction(
    wavelength_um: np.ndarray, ebv: float, rv: float
) -> np.ndarray:
    """Compute CCM89 extinction correction factor (O'Donnell 1994 update)."""

    x = 1.0 / np.clip(wavelength_um, 1e-6, None)  # inverse micron
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # Infrared (0.3 <= x < 1.1 μm^-1)
    mask_ir = (x >= 0.3) & (x < 1.1)
    if np.any(mask_ir):
        y = x[mask_ir]
        a[mask_ir] = 0.574 * y ** 1.61
        b[mask_ir] = -0.527 * y ** 1.61

    # Optical/NIR (1.1 <= x < 3.3 μm^-1)
    mask_opt = (x >= 1.1) & (x < 3.3)
    if np.any(mask_opt):
        y = x[mask_opt] - 1.82
        a_coeffs = [1.0, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999]
        b_coeffs = [0.0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002]
        a[mask_opt] = np.polyval(list(reversed(a_coeffs)), y)
        b[mask_opt] = np.polyval(list(reversed(b_coeffs)), y)

    # UV (3.3 <= x <= 8)
    mask_uv = (x >= 3.3) & (x <= 8.0)
    if np.any(mask_uv):
        y = x[mask_uv]
        Fa = np.zeros_like(y)
        Fb = np.zeros_like(y)
        mask_far_uv = y > 5.9
        if np.any(mask_far_uv):
            yfu = y[mask_far_uv] - 5.9
            Fa[mask_far_uv] = -0.04473 * yfu ** 2 - 0.009779 * yfu ** 3
            Fb[mask_far_uv] = 0.2130 * yfu ** 2 + 0.1207 * yfu ** 3
        a[mask_uv] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341) + Fa
        b[mask_uv] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263) + Fb

    Alambda = (a + b / rv) * ebv * rv
    factor = 10 ** (0.4 * Alambda)
    return factor


def apply_dereddening(
    data: SpectrumData,
    do_deredden: bool,
    ra: float,
    dec: float,
    law: str,
    rv: float,
    module_path: Path,
    readme: ReadmeLogger,
) -> DereddeningResult:
    notes: List[str] = []
    if not do_deredden:
        notes.append("Dereddening disabled by user option.")
        return DereddeningResult(data.flux, data.error, np.ones_like(data.flux), None, None, None, notes)

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
    except Exception as exc:
        msg = f"astropy not available ({exc}); skipping dereddening."
        logging.warning(msg)
        readme.add(msg)
        notes.append(msg)
        return DereddeningResult(data.flux, data.error, np.ones_like(data.flux), None, None, None, notes)

    try:
        from dustmaps.sfd import SFDQuery
    except Exception as exc:
        msg = f"dustmaps.sfd not available ({exc}); skipping dereddening."
        logging.warning(msg)
        readme.add(msg)
        notes.append(msg)
        return DereddeningResult(data.flux, data.error, np.ones_like(data.flux), None, None, None, notes)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    try:
        sfd = SFDQuery()
        ebv = float(sfd(coord))
        logging.info("Queried SFD E(B-V)=%.4f at RA=%.5f DEC=%.5f", ebv, ra, dec)
        readme.add(
            f"Dereddening enabled: E(B-V)={ebv:.4f} at RA={ra:.5f} DEC={dec:.5f} using SFD map"
        )
    except Exception as exc:
        msg = (
            "Unable to query SFD dust map; ensure dustmaps data is installed. "
            f"Skipping dereddening ({exc})."
        )
        logging.warning(msg)
        readme.add(msg)
        notes.append(msg)
        return DereddeningResult(data.flux, data.error, np.ones_like(data.flux), None, None, None, notes)

    module = import_extinction_module(module_path, readme)
    factor: Optional[np.ndarray] = None

    if module is not None:
        func = find_extinction_function(module)
        if func is not None:
            try:
                result = func(
                    data.wavelength_um,
                    data.flux,
                    ebv=ebv,
                    law=law,
                    rv=rv,
                )
                if isinstance(result, tuple):
                    if len(result) == 2:
                        corrected_flux, factor = result
                    elif len(result) >= 3:
                        corrected_flux, _, factor = result[:3]
                    else:
                        corrected_flux = result[0]
                elif isinstance(result, dict):
                    corrected_flux = result.get("flux", data.flux * 1.0)
                    factor = result.get("factor")
                else:
                    corrected_flux = np.asarray(result)

                if factor is None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        factor = np.where(data.flux != 0, corrected_flux / data.flux, 1.0)
                logging.info("Applied extinction correction via external module.")
            except Exception as exc:
                msg = f"Extinction module failed ({exc}); falling back to internal CCM89 implementation."
                logging.warning(msg)
                readme.add(msg)
                notes.append(msg)
                factor = None
        else:
            msg = "Extinction module lacks a known correction function; using internal CCM89 implementation."
            logging.warning(msg)
            readme.add(msg)
            notes.append(msg)

    if factor is None:
        if law != "ccm89":
            msg = f"Internal implementation currently supports only CCM89; requested {law}."
            logging.warning(msg)
            readme.add(msg)
            notes.append(msg)
        factor = compute_ccm89_correction(data.wavelength_um, ebv, rv)
        corrected_flux = data.flux * factor

    corrected_error = data.error * factor

    if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
        msg = "Non-finite or non-positive extinction factors detected; reverting to raw flux."
        logging.warning(msg)
        readme.add(msg)
        notes.append(msg)
        return DereddeningResult(data.flux, data.error, np.ones_like(data.flux), ebv, law, rv, notes)

    readme.add(f"Applied extinction law={law}, R_V={rv:.2f} (factor median={float(np.median(factor)):.3f})")
    return DereddeningResult(corrected_flux, corrected_error, factor, ebv, law, rv, notes)


def plot_dereddened_comparison(
    data: SpectrumData,
    dered: DereddeningResult,
    out_path: Path,
) -> None:
    logging.info("Creating dereddened comparison plot: %s", out_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(data.wavelength_um, data.flux, "o", ms=3, alpha=0.6, label="Raw")
    if not np.allclose(dered.factor, 1.0):
        ax.plot(data.wavelength_um, dered.flux, "-", lw=1.5, alpha=0.8, label="Dereddened")
    else:
        ax.plot(data.wavelength_um, dered.flux, "-", lw=1.2, alpha=0.8, label="Processed")
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"F$_{\lambda}$ [cgs]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ebv_str = f"E(B-V)={dered.ebv:.3f}" if dered.ebv is not None else "E(B-V)=N/A"
    law_str = dered.law or "N/A"
    rv_str = f"R_V={dered.rv:.2f}" if dered.rv is not None else "R_V=N/A"
    ax.set_title("Dereddening Comparison")
    ax.legend(title=f"{ebv_str}\nLaw={law_str}, {rv_str}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def compute_sn_statistics(flux: np.ndarray, error: np.ndarray) -> Dict[str, float]:
    sn = flux / error
    finite = np.isfinite(sn)
    valid_sn = sn[finite]
    stats = {
        "count": int(valid_sn.size),
        "mean": float(np.mean(valid_sn)) if valid_sn.size else math.nan,
        "median": float(np.median(valid_sn)) if valid_sn.size else math.nan,
        "std": float(np.std(valid_sn)) if valid_sn.size else math.nan,
        "min": float(np.min(valid_sn)) if valid_sn.size else math.nan,
        "max": float(np.max(valid_sn)) if valid_sn.size else math.nan,
    }
    return stats


def plot_sn_histogram(sn_values: np.ndarray, out_path: Path) -> None:
    logging.info("Creating S/N histogram: %s", out_path)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    finite = np.isfinite(sn_values)
    ax.hist(sn_values[finite], bins=30, color="#4c72b0", alpha=0.8)
    ax.set_xlabel("Signal-to-Noise (F_lambda / eF_lambda)")
    ax.set_ylabel("Counts")
    ax.set_title("S/N Distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_prospector_obs(
    wavelengths_um: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    readme: ReadmeLogger,
) -> None:
    logging.info("Saving Prospector observation dictionary to %s", out_path)
    wavelengths_A = wavelengths_um * 1e4
    obs = {
        "wavelength": wavelengths_A,
        "spectrum": flux,
        "unc": error,
        "mask": mask,
        "redshift": 0.0,
        "filters": None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **obs)

    # Self-check: load back and verify shapes.
    loaded = np.load(out_path, allow_pickle=True)
    keys = set(loaded.keys())
    expected_keys = {"wavelength", "spectrum", "unc", "mask", "redshift", "filters"}
    if keys != expected_keys:
        msg = f"Prospector obs saved keys mismatch: {keys}"
        logging.warning(msg)
        readme.add(msg)
    for key in ["wavelength", "spectrum", "unc", "mask"]:
        arr = loaded[key]
        if arr.shape != flux.shape:
            msg = f"Prospector obs {key} shape mismatch: {arr.shape} vs {flux.shape}"
            logging.warning(msg)
            readme.add(msg)
    loaded.close()
    readme.add("Prospector obs dictionary saved and validated for expected keys/shapes.")


def run_quickfit(
    wavelengths_A: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    out_path: Path,
    readme: ReadmeLogger,
) -> None:
    logging.info("Attempting quick Prospector SSP fit.")
    try:
        from prospect.models.templates import TemplateLibrary
        from prospect.fitting import fit_model
        from prospect.models import model_setup
        import fsps  # noqa: F401 (ensure availability)
    except Exception as exc:
        msg = f"Prospector/FSPS dependencies unavailable: {exc}. Skipping quick fit."
        logging.warning(msg)
        readme.add(msg)
        return

    try:
        model_params = TemplateLibrary["ssp"]()
        model_params.update({
            "logzsol": {"N": 1, "isfree": True, "init": -0.2, "prior": {"name": "uniform", "limits": (-1.0, 0.3)}},
            "tage": {"N": 1, "isfree": True, "init": 8.0, "prior": {"name": "uniform", "limits": (0.1, 13.5)}},
            "dust2": {"N": 1, "isfree": True, "init": 0.1, "prior": {"name": "uniform", "limits": (0.0, 2.0)}},
        })
        obs = {
            "wavelength": wavelengths_A,
            "spectrum": flux,
            "unc": error,
            "mask": np.isfinite(flux) & np.isfinite(error) & (error > 0),
            "filters": None,
            "redshift": 0.0,
        }
        model = model_setup.setup_model(model_params=model_params)["model"]

        dynesty_opt = {
            "nlive_init": 150,
            "maxcall": 4000,
        }
        output = fit_model(obs, model=model, dynesty_opt=dynesty_opt)
        logs = output["sampling_results"]
        weights = logs["weights"]
        logl = logs["logl"]
        best_idx = np.argmax(logl)
        best_spectrum = model.mean_model(
            logs["samples"][best_idx],
            obs,
            sps=model_setup.setup_sps()["sps"],
        )
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        ax.plot(wavelengths_A, flux, label="Data", lw=1.0)
        ax.plot(wavelengths_A, best_spectrum, label="Best SSP", lw=1.2)
        ax.fill_between(
            wavelengths_A,
            flux - error,
            flux + error,
            color="#cccccc",
            alpha=0.4,
            label="±1σ",
        )
        ax.set_xlabel(r"Wavelength [Å]")
        ax.set_ylabel(r"F$_{\lambda}$ [cgs]")
        ax.set_title("Quick Prospector SSP Fit")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

        max_logl = float(np.max(logl))
        med_params = np.median(logs["samples"], axis=0)
        readme.add(
            "Quickfit completed. Max log-likelihood = {:.2f}; median parameters = {}".format(
                max_logl, np.array2string(med_params, precision=3)
            )
        )
    except Exception as exc:
        msg = f"Quick Prospector fit failed: {exc}"
        logging.warning(msg)
        readme.add(msg)


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)

    readme = ReadmeLogger()
    readme.add(f"Run timestamp: {datetime.now(timezone.utc).isoformat()}")
    readme.add(f"Command: {' '.join([sys.executable] + sys.argv)}")
    readme.add(
        json.dumps(
            {
                "csv": args.csv,
                "ext": args.ext,
                "ra": args.ra,
                "dec": args.dec,
                "deredden": args.deredden,
                "law": args.law,
                "rv": args.rv,
                "outdir": args.outdir,
                "quickfit": args.quickfit,
            },
            indent=2,
        )
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_spectrum(Path(args.csv), readme)
        raw_df = pd.read_csv(args.csv)
        save_head_summary(raw_df, outdir / "00_head.csv")
    except Exception as exc:
        logging.error("Failed to load spectrum: %s", exc)
        readme.add(f"Fatal error during CSV loading: {exc}")
        final_path = outdir / "README.txt"
        final_path.write_text(readme.render(), encoding="utf-8")
        raise

    plot_raw_spectrum(data, outdir / "10_qc_raw.png")

    deredden_choice = args.deredden.lower() == "true"
    dered = apply_dereddening(
        data,
        deredden_choice,
        args.ra,
        args.dec,
        args.law,
        args.rv,
        Path(args.ext),
        readme,
    )
    if dered.notes:
        readme.extend(dered.notes)

    plot_dereddened_comparison(data, dered, outdir / "20_dereddened.png")

    sn_raw = data.flux / data.error
    sn_dered = dered.flux / dered.error
    sn_stats_raw = compute_sn_statistics(data.flux, data.error)
    sn_stats_dered = compute_sn_statistics(dered.flux, dered.error)
    logging.info(
        "S/N raw mean/median=%.2f/%.2f; dered mean/median=%.2f/%.2f",
        sn_stats_raw["mean"],
        sn_stats_raw["median"],
        sn_stats_dered["mean"],
        sn_stats_dered["median"],
    )
    readme.add("S/N stats (raw): " + json.dumps(sn_stats_raw, indent=2))
    readme.add("S/N stats (dereddened): " + json.dumps(sn_stats_dered, indent=2))
    plot_sn_histogram(sn_dered if dered.ebv is not None else sn_raw, outdir / "30_sn_hist.png")

    mask = np.isfinite(dered.flux) & np.isfinite(dered.error) & (dered.error > 0)
    readme.add(f"Prospector mask valid points: {int(np.count_nonzero(mask))} / {mask.size}")
    save_prospector_obs(
        data.wavelength_um,
        dered.flux if dered.ebv is not None else data.flux,
        dered.error if dered.ebv is not None else data.error,
        mask,
        outdir / "40_obs_dict.npz",
        readme,
    )

    if args.quickfit.lower() == "true":
        run_quickfit(data.wavelength_um * 1e4, dered.flux, dered.error, outdir / "50_quickfit.png", readme)
    else:
        readme.add("Quickfit skipped (option set to false).")

    readme.add("Output files:")
    for filename in [
        "00_head.csv",
        "10_qc_raw.png",
        "20_dereddened.png",
        "30_sn_hist.png",
        "40_obs_dict.npz",
    ]:
        path = outdir / filename
        if path.exists():
            readme.add(f" - {path}")
    if (outdir / "50_quickfit.png").exists():
        readme.add(f" - {outdir / '50_quickfit.png'}")

    readme_path = outdir / "README.txt"
    readme_path.write_text(readme.render(), encoding="utf-8")
    logging.info("Analysis complete. README saved to %s", readme_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        logging.error("Unhandled exception: %s", err)
        sys.exit(1)
