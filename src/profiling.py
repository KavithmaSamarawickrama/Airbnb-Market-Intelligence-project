"""
Data quality profiling and validation.

Computes null rates, cardinality, outliers, and data quality metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import yaml

logger = logging.getLogger(__name__)


class DataQualityProfiler:
    """Profiles data quality across Bronze and Silver layers."""

    def __init__(self, settings_path: Path):
        """
        Initialize profiler.

        Args:
            settings_path: Path to settings.yaml
        """
        self.settings_path = Path(settings_path)
        with open(self.settings_path) as f:
            self.settings = yaml.safe_load(f)

        db_file = self.settings["duckdb"]["database_file"]
        self.conn = duckdb.connect(db_file)

    def profile_table(self, table_path: str, table_name: str) -> Dict:
        """
        Profile a table (Parquet file).

        Returns:
            Dict with null_rates, cardinality, etc.
        """
        try:
            logger.info(f"Profiling {table_name}...")

            # Get schema
            schema_query = f"DESCRIBE read_parquet('{table_path}')"
            schema = self.conn.execute(schema_query).fetchall()

            profile = {
                "table": table_name,
                "path": table_path,
                "columns": [],
            }

            # Get row count
            row_count = self.conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{table_path}')"
            ).fetchone()[0]
            profile["total_rows"] = row_count

            # Profile each column
            for col_name, col_type in schema:
                col_profile = {
                    "name": col_name,
                    "type": col_type,
                }

                # Null rate
                null_query = (
                    f"SELECT COUNT(*) FROM read_parquet('{table_path}') "
                    f"WHERE {col_name} IS NULL"
                )
                null_count = self.conn.execute(null_query).fetchone()[0]
                col_profile["null_count"] = null_count
                col_profile["null_rate"] = null_count / row_count if row_count > 0 else 0

                # Cardinality (distinct values)
                card_query = (
                    f"SELECT COUNT(DISTINCT {col_name}) FROM read_parquet('{table_path}')"
                )
                cardinality = self.conn.execute(card_query).fetchone()[0]
                col_profile["cardinality"] = cardinality

                # Numeric stats (if numeric)
                if "INT" in col_type.upper() or "FLOAT" in col_type.upper():
                    stats_query = (
                        f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}), "
                        f"STDDEV({col_name}) FROM read_parquet('{table_path}') "
                        f"WHERE {col_name} IS NOT NULL"
                    )
                    min_val, max_val, avg_val, stddev_val = self.conn.execute(
                        stats_query
                    ).fetchone()
                    col_profile["min"] = min_val
                    col_profile["max"] = max_val
                    col_profile["avg"] = avg_val
                    col_profile["stddev"] = stddev_val

                profile["columns"].append(col_profile)

            logger.info(f"Profiled {table_name}: {row_count} rows, {len(schema)} columns")
            return profile

        except Exception as e:
            logger.error(f"Failed to profile {table_name}: {e}")
            return {"table": table_name, "error": str(e)}

    def validate_data(self, profile: Dict) -> List[str]:
        """
        Validate data based on quality thresholds from settings.

        Returns:
            List of validation warnings/errors
        """
        issues = []
        quality = self.settings.get("quality", {})

        row_count = profile.get("total_rows", 0)
        if row_count < quality.get("min_rows", 10):
            issues.append(f"Table has only {row_count} rows (min: {quality['min_rows']})")

        max_null_frac = quality.get("max_null_fraction", 0.5)
        for col in profile.get("columns", []):
            null_rate = col.get("null_rate", 0)
            if null_rate > max_null_frac:
                issues.append(
                    f"Column {col['name']} has null rate {null_rate:.2%} (max: {max_null_frac:.2%})"
                )

        return issues

    def run(self, cities: List[str]) -> Dict:
        """
        Execute profiling for all cities.

        Args:
            cities: List of city names

        Returns:
            Profiling report dict
        """
        report = {"timestamp": str(Path.cwd()), "profiles": []}

        silver_dir = Path("./data/processed_silver")

        for city in cities:
            logger.info(f"Profiling {city}...")

            listings_path = silver_dir / f"{city}_listings_clean.parquet"
            if listings_path.exists():
                profile = self.profile_table(str(listings_path), f"{city}_listings")
                issues = self.validate_data(profile)
                profile["validation_issues"] = issues
                report["profiles"].append(profile)

            calendar_path = silver_dir / f"{city}_calendar_clean.parquet"
            if calendar_path.exists():
                profile = self.profile_table(str(calendar_path), f"{city}_calendar")
                issues = self.validate_data(profile)
                profile["validation_issues"] = issues
                report["profiles"].append(profile)

            reviews_path = silver_dir / f"{city}_reviews_clean.parquet"
            if reviews_path.exists():
                profile = self.profile_table(str(reviews_path), f"{city}_reviews")
                issues = self.validate_data(profile)
                profile["validation_issues"] = issues
                report["profiles"].append(profile)

        logger.info(f"Profiling complete. Found {len(report['profiles'])} tables.")
        return report

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


def run(settings_path: str = "./config/settings.yaml", cities: Optional[List[str]] = None) -> Dict:
    """
    Entry point for profiling.

    Args:
        settings_path: Path to settings.yaml
        cities: List of cities to profile

    Returns:
        Profiling report dict
    """
    if cities is None:
        cities = ["london", "paris"]

    profiler = DataQualityProfiler(settings_path)
    try:
        return profiler.run(cities)
    finally:
        profiler.close()


if __name__ == "__main__":
    import json
    import sys
    from utils import setup_logging

    setup_logging(Path("./logs"), level="INFO", format_type="json")
    result = run()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)
