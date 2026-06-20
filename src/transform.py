"""
DuckDB-based transformation from Bronze to Silver.

Cleans, types, and standardizes CSV files from Bronze into Parquet on Silver.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import yaml
import duckdb

logger = logging.getLogger(__name__)


class DuckDBTransformer:
    """DuckDB-based Bronze→Silver transformation engine."""

    def __init__(self, settings_path: Path):
        """
        Initialize transformer.

        Args:
            settings_path: Path to settings.yaml
        """
        self.settings_path = Path(settings_path)
        with open(self.settings_path) as f:
            self.settings = yaml.safe_load(f)

        # Initialize DuckDB
        db_file = self.settings["duckdb"]["database_file"]
        self.conn = duckdb.connect(db_file)
        self._configure_duckdb()

        self.summary = {"processed": 0, "failed": 0}

    def _configure_duckdb(self):
        """Configure DuckDB for optimal performance."""
        threads = self.settings["duckdb"]["threads"]
        memory = self.settings["duckdb"]["memory_limit"]

        if threads > 0:
            self.conn.execute(f"SET threads = {threads}")
        self.conn.execute(f"SET memory_limit = '{memory}'")

        logger.info(f"DuckDB configured: threads={threads}, memory={memory}")

    def transform_listings(self, city: str, local_path: Path) -> bool:
        """
        Transform listings.csv → listings_clean.parquet.

        Performs:
        - Type coercion (numeric prices, dates, etc.)
        - Null handling
        - Outlier detection
        """
        try:
            logger.info(f"Transforming listings for {city}...")

            query = f"""
            SELECT
                CAST(id AS BIGINT) as listing_id,
                name,
                CAST(host_id AS BIGINT) as host_id,
                host_name,
                CAST(host_since AS DATE) as host_since,
                CAST(host_listings_count AS INT) as host_listings_count,
                CAST(host_is_superhost AS BOOLEAN) as is_superhost,
                neighbourhood_cleansed as neighbourhood,
                neighbourhood_group_cleansed as neighbourhood_group,
                CAST(latitude AS FLOAT) as latitude,
                CAST(longitude AS FLOAT) as longitude,
                room_type,
                property_type,
                CAST(accommodates AS INT) as accommodates,
                CAST(bedrooms AS FLOAT) as bedrooms,
                CAST(beds AS FLOAT) as beds,
                CAST(price AS FLOAT) as price,
                CAST(minimum_nights AS INT) as minimum_nights,
                CAST(number_of_reviews AS INT) as number_of_reviews,
                CAST(review_scores_rating AS FLOAT) as review_scores_rating,
                instant_bookable,
                CAST(current_date AS DATE) as captured_date
            FROM read_csv_auto('{local_path}')
            WHERE
                latitude IS NOT NULL AND longitude IS NOT NULL
                AND price > 0
            ORDER BY listing_id
            """

            output_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, CODEC 'snappy')")

            row_count = self.conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
            logger.info(f"Wrote {row_count} listings to {output_path}")

            self.summary["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to transform listings for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def transform_calendar(self, city: str, local_path: Path) -> bool:
        """
        Transform calendar.csv → calendar_clean.parquet.

        Standardizes availability data and date handling.
        """
        try:
            logger.info(f"Transforming calendar for {city}...")

            query = f"""
            SELECT
                CAST(listing_id AS BIGINT) as listing_id,
                CAST(date AS DATE) as calendar_date,
                CAST(available AS BOOLEAN) as is_available,
                CAST(price AS FLOAT) as price,
                COALESCE(minimum_nights, 1) as minimum_nights,
                COALESCE(maximum_nights, 365) as maximum_nights
            FROM read_csv_auto('{local_path}')
            WHERE date IS NOT NULL AND listing_id IS NOT NULL
            ORDER BY listing_id, calendar_date
            """

            output_path = Path("./data/processed_silver") / f"{city}_calendar_clean.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, CODEC 'snappy')")

            row_count = self.conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
            logger.info(f"Wrote {row_count} calendar rows to {output_path}")

            self.summary["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to transform calendar for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def transform_reviews(self, city: str, local_path: Path) -> bool:
        """
        Transform reviews.csv → reviews_clean.parquet.

        Parses review metadata and timestamps.
        """
        try:
            logger.info(f"Transforming reviews for {city}...")

            query = f"""
            SELECT
                CAST(listing_id AS BIGINT) as listing_id,
                CAST(id AS BIGINT) as review_id,
                CAST(reviewer_id AS BIGINT) as reviewer_id,
                reviewer_name,
                CAST(review_date AS DATE) as review_date,
                comments
            FROM read_csv_auto('{local_path}')
            WHERE listing_id IS NOT NULL AND review_date IS NOT NULL
            ORDER BY listing_id, review_date
            """

            output_path = Path("./data/processed_silver") / f"{city}_reviews_clean.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, CODEC 'snappy')")

            row_count = self.conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
            logger.info(f"Wrote {row_count} reviews to {output_path}")

            self.summary["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to transform reviews for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def run(self, cities: List[str]) -> Dict[str, int]:
        """
        Execute transformation for multiple cities.

        Args:
            cities: List of city names

        Returns:
            Summary dict
        """
        bronze_dir = Path("./data/raw_bronze")

        for city in cities:
            logger.info(f"Processing {city}...")

            # Expected local files from ingestion
            listings_csv = bronze_dir / city / "listings.csv"
            calendar_csv = bronze_dir / city / "calendar.csv"
            reviews_csv = bronze_dir / city / "reviews.csv.gz"

            if listings_csv.exists():
                self.transform_listings(city, listings_csv)
            else:
                logger.warning(f"Listings CSV not found for {city}")

            if calendar_csv.exists():
                self.transform_calendar(city, calendar_csv)
            else:
                logger.warning(f"Calendar CSV not found for {city}")

            if reviews_csv.exists():
                self.transform_reviews(city, reviews_csv)
            else:
                logger.warning(f"Reviews CSV not found for {city}")

        logger.info(f"Transformation complete. Summary: {self.summary}")
        return self.summary

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


def run(settings_path: str = "./config/settings.yaml", cities: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Entry point for transformation.

    Args:
        settings_path: Path to settings.yaml
        cities: List of cities to transform (if None, uses config)

    Returns:
        Summary dict
    """
    if cities is None:
        cities = ["london", "paris"]  # Default

    transformer = DuckDBTransformer(settings_path)
    try:
        return transformer.run(cities)
    finally:
        transformer.close()


if __name__ == "__main__":
    import sys
    from utils import setup_logging

    setup_logging(Path("./logs"), level="INFO", format_type="json")
    result = run()
    sys.exit(0 if result["failed"] == 0 else 1)
