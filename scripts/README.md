# scripts/

Small operational/utility scripts (data refresh, report generation, etc.).
Scripts should be thin wrappers around functions in `src/strategy_lab/` rather
than containing their own business logic, so that logic stays testable.
