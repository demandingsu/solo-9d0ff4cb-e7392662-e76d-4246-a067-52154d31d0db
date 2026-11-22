"""Domain exceptions used across the reconciliation service."""


class ReconciliationError(Exception):
    """Base class for domain-level failures that should be shown to users."""

    code = "reconciliation_error"


class CsvParseError(ReconciliationError):
    code = "csv_parse_error"


class MappingValidationError(ReconciliationError):
    """The uploaded tree mapping is not a valid forest for reconciliation."""

    code = "mapping_validation_error"


class SnapshotValidationError(ReconciliationError):
    """A submission is internally inconsistent or incomplete."""

    code = "snapshot_validation_error"


class OptimizationError(ReconciliationError):
    """The weighted least-squares correction could not be solved."""

    code = "optimization_error"


class StalePreviewError(ReconciliationError):
    """A preview belongs to a superseded input version and may not be published."""

    code = "stale_preview_error"


class PublishError(ReconciliationError):
    code = "publish_error"
