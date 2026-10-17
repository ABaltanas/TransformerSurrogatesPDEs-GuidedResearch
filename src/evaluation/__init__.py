"""Surrogate evaluation and scientific benchmarking subsystem."""

from src.evaluation.denormalizer import StatisticalDenormalizer
from src.evaluation.engine import EvaluationEngine
from src.evaluation.exporters import (
    CompositeReportExporter,
    JsonReportExporter,
    MarkdownReportExporter,
)
from src.evaluation.interfaces import (
    BenchmarkProfiler,
    Denormalizer,
    FeasibilityValidator,
    InferenceRunner,
    PhysicalMetricsCalculator,
    ReportExporter,
    TrainingHistoryAnalyzer,
)
from src.evaluation.metrics import PlasticityMetricsCalculator
from src.evaluation.profiler import HardwareProfiler
from src.evaluation.runner import BatchedInferenceRunner
from src.evaluation.training_history import TrainingTelemetryAnalyzer
from src.evaluation.validators import DiePenetrationValidator

__all__ = [
    # Interfaces
    "InferenceRunner",
    "Denormalizer",
    "PhysicalMetricsCalculator",
    "FeasibilityValidator",
    "BenchmarkProfiler",
    "ReportExporter",
    "TrainingHistoryAnalyzer",
    # Implementations
    "BatchedInferenceRunner",
    "StatisticalDenormalizer",
    "PlasticityMetricsCalculator",
    "DiePenetrationValidator",
    "HardwareProfiler",
    "MarkdownReportExporter",
    "JsonReportExporter",
    "CompositeReportExporter",
    "TrainingTelemetryAnalyzer",
    # Orchestrator
    "EvaluationEngine",
]

