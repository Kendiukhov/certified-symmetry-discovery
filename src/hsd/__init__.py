"""Statistically honest symmetry discovery from noisy, limited data."""

from .polynomials import PolyAlgebra, bracket, jac_times, moment_matrix, poly_algebra
from .defect import DefectProblem
from .systems import SYSTEMS, System, exact_symmetry_algebra, get_system
from .estimation import Fit, fit_ols, region_covers, simulate_design
from .certify import (Certificate, certified_dimension, certify_direction,
                      certify_subspace, refute_all)
from .coverage import coverage_report, extrapolation_factor
from . import baselines, closure, dictionary, selection, trajectories

__version__ = "2.0.0"

__all__ = [
    "PolyAlgebra", "bracket", "jac_times", "moment_matrix", "poly_algebra",
    "DefectProblem", "SYSTEMS", "System", "exact_symmetry_algebra", "get_system",
    "Fit", "fit_ols", "region_covers", "simulate_design", "Certificate",
    "certified_dimension", "closure", "dictionary", "selection", "trajectories",
    "certify_direction", "certify_subspace", "refute_all", "coverage_report",
    "extrapolation_factor", "baselines", "__version__",
]
