"""Emission-density families."""

from hsmmkit.model.emission.base import EmissionFamily, Support
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.emission.student_t import StudentTFamily

__all__ = ["EmissionFamily", "GaussianFamily", "ProductEmission", "StudentTFamily", "Support"]
