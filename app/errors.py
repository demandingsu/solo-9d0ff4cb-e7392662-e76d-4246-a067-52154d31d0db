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


class LineLimitExceededError(ReconciliationError):
    """Some periods need the dense solver but the line count exceeds the limit.

    Raised *before* any period is solved: the whole batch is rejected and no
    preview is stored, so the last published snapshot keeps serving data.
    """

    code = "line_limit_exceeded"

    def __init__(self, periods: list[str], n_lines: int, limit: int) -> None:
        self.periods = list(periods)
        self.n_lines = n_lines
        self.limit = limit
        shown = ", ".join(self.periods[:12])
        more = f" 等 {len(self.periods)} 个时段" if len(self.periods) > 12 else ""
        super().__init__(
            f"以下时段含产线级发布值或产线权重为 0，需使用通用求解器，"
            f"但产线数 {n_lines} 超过上限 {limit}：{shown}{more}。"
            f"请将发布值改到车间/工厂层级、确保产线权重大于 0，"
            f"或调大 RECONCILE_MAX_LINES 后重试。"
        )

    @property
    def detail(self) -> dict:
        return {"periods": list(self.periods), "n_lines": self.n_lines, "limit": self.limit}


class StalePreviewError(ReconciliationError):
    """A preview belongs to a superseded input version and may not be published."""

    code = "stale_preview_error"


class PublishError(ReconciliationError):
    code = "publish_error"
