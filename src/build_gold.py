"""
Star schema construction and Gold layer load.

Reads Silver Parquet files, builds conformed dimensions and fact tables,
and loads them into Supabase (managed Postgres) free tier.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import yaml
import duckdb
import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)


class GoldLayerBuilder:
    """Builds and loads Gold star schema to Supabase."""

    def __init__(self, settings_path: Path):
        """
        Initialize builder.

        Args:
            settings_path: Path to settings.yaml
        """
        self.settings_path = Path(settings_path)
        with open(self.settings_path) as f:
            self.settings = yaml.safe_load(f)

        # Initialize DuckDB
        db_file = self.settings["duckdb"]["database_file"]
        self.duck = duckdb.connect(db_file)
        self._configure_duckdb()

        # Supabase connection (lazy load)
        self.pg_conn = None
        self.summary = {"dimensions": 0, "fact": 0, "failed": 0}

    def _configure_duckdb(self):
        """Configure DuckDB for optimal performance."""
        threads = self.settings["duckdb"]["threads"]
        memory = self.settings["duckdb"]["memory_limit"]
        if threads > 0:
            self.duck.execute(f"SET threads = {threads}")
        self.duck.execute(f"SET memory_limit = '{memory}'")

    def _get_pg_conn(self):
        """Get or create Postgres connection."""
        if self.pg_conn is None:
            try:
                sb = self.settings["supabase"]
                self.pg_conn = psycopg2.connect(
                    host=sb["host"],
                    port=sb["port"],
                    database=sb["database"],
                    user=sb["user"],
                    password=sb["password"],
                )
                logger.info(f"Connected to Supabase at {sb['host']}")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                raise
        return self.pg_conn

    def build_dim_calendar(self, start_date: str = "2020-01-01", end_date: str = "2030-12-31") -> bool:
        """
        Build dim_calendar_dates from date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if successful
        """
        try:
            logger.info("Building dim_calendar_dates...")

            query = f"""
            CREATE OR REPLACE TABLE dim_calendar_dates AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY full_date) as date_key,
                full_date,
                YEAR(full_date) as year,
                MONTH(full_date) as month,
                DAY_OF_WEEK(full_date) as day_of_week,
                (DAY_OF_WEEK(full_date) IN (6, 7)) as is_weekend,
                CASE
                    WHEN MONTH(full_date) IN (12, 1, 2) THEN 'Winter'
                    WHEN MONTH(full_date) IN (3, 4, 5) THEN 'Spring'
                    WHEN MONTH(full_date) IN (6, 7, 8) THEN 'Summer'
                    ELSE 'Fall'
                END as season
            FROM (
                SELECT CAST(d AS DATE) as full_date
                FROM generate_series('{start_date}'::DATE, '{end_date}'::DATE, '1 day'::INTERVAL) AS t(d)
            )
            ORDER BY full_date
            """

            self.duck.execute(query)
            row_count = self.duck.execute("SELECT COUNT(*) FROM dim_calendar_dates").fetchone()[0]
            logger.info(f"Created dim_calendar_dates with {row_count} rows")
            self.summary["dimensions"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to build dim_calendar_dates: {e}")
            self.summary["failed"] += 1
            return False

    def build_dim_listings(self, city: str) -> bool:
        """Build dim_listings from Silver listings_clean."""
        try:
            logger.info(f"Building dim_listings for {city}...")

            # Read from Silver Parquet
            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_listings_{city} AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY listing_id) as listing_key,
                listing_id,
                name,
                room_type,
                property_type,
                accommodates,
                bedrooms,
                beds,
                latitude,
                longitude
            FROM read_parquet('{listings_path}')
            ORDER BY listing_id
            """

            self.duck.execute(query)
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_listings_{city}").fetchone()[0]
            logger.info(f"Created dim_listings_{city} with {row_count} rows")
            self.summary["dimensions"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to build dim_listings for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def build_dim_hosts(self, city: str) -> bool:
        """Build dim_hosts from Silver listings_clean."""
        try:
            logger.info(f"Building dim_hosts for {city}...")

            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_hosts_{city} AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY host_id) as host_key,
                host_id,
                host_name,
                is_superhost,
                host_since,
                host_listings_count
            FROM (
                SELECT DISTINCT
                    host_id,
                    host_name,
                    is_superhost,
                    host_since,
                    host_listings_count
                FROM read_parquet('{listings_path}')
                WHERE host_id IS NOT NULL
            )
            ORDER BY host_id
            """

            self.duck.execute(query)
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_hosts_{city}").fetchone()[0]
            logger.info(f"Created dim_hosts_{city} with {row_count} rows")
            self.summary["dimensions"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to build dim_hosts for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def build_dim_neighbourhoods(self, city: str) -> bool:
        """Build dim_neighbourhoods from Silver listings_clean."""
        try:
            logger.info(f"Building dim_neighbourhoods for {city}...")

            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_neighbourhoods_{city} AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY neighbourhood) as neighbourhood_key,
                neighbourhood,
                neighbourhood_group,
                AVG(latitude) as centroid_lat,
                AVG(longitude) as centroid_lon
            FROM (
                SELECT DISTINCT
                    neighbourhood,
                    neighbourhood_group,
                    latitude,
                    longitude
                FROM read_parquet('{listings_path}')
                WHERE neighbourhood IS NOT NULL
            )
            GROUP BY neighbourhood, neighbourhood_group
            ORDER BY neighbourhood
            """

            self.duck.execute(query)
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_neighbourhoods_{city}").fetchone()[0]
            logger.info(f"Created dim_neighbourhoods_{city} with {row_count} rows")
            self.summary["dimensions"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to build dim_neighbourhoods for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def build_fact_listings_daily(self, city: str) -> bool:
        """
        Build fact_listings_daily_snapshot.

        Grain: one row per listing per calendar date.
        Source: calendar + listings (left join).
        """
        try:
            logger.info(f"Building fact_listings_daily_snapshot for {city}...")

            calendar_path = Path("./data/processed_silver") / f"{city}_calendar_clean.parquet"
            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"

            for path in [calendar_path, listings_path]:
                if not path.exists():
                    logger.warning(f"Silver file not found: {path}")
                    return False

            # Complex join with dimension tables
            query = f"""
            CREATE OR REPLACE TABLE fact_listings_daily_{city} AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY dl.listing_key, dc.date_key) as fact_id,
                dl.listing_key,
                dh.host_key,
                dn.neighbourhood_key,
                dc.date_key,
                cal.price,
                cal.is_available,
                cal.minimum_nights,
                COALESCE(cal.price * (CASE WHEN cal.is_available THEN 1.0 ELSE 0.0 END), 0) as est_revenue,
                COALESCE(lr.review_count, 0) as reviews_to_date
            FROM read_parquet('{calendar_path}') as cal
            LEFT JOIN read_parquet('{listings_path}') as lst
                ON cal.listing_id = lst.listing_id
            LEFT JOIN dim_listings_{city} dl
                ON lst.listing_id = dl.listing_id
            LEFT JOIN dim_hosts_{city} dh
                ON lst.host_id = dh.host_id
            LEFT JOIN dim_neighbourhoods_{city} dn
                ON lst.neighbourhood = dn.neighbourhood
            LEFT JOIN dim_calendar_dates dc
                ON CAST(cal.calendar_date AS DATE) = dc.full_date
            LEFT JOIN (
                SELECT listing_id, COUNT(*) as review_count
                FROM read_parquet('{Path("./data/processed_silver") / f"{city}_reviews_clean.parquet"}')
                GROUP BY listing_id
            ) lr ON cal.listing_id = lr.listing_id
            WHERE dl.listing_key IS NOT NULL
            ORDER BY dl.listing_key, dc.date_key
            """

            self.duck.execute(query)
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM fact_listings_daily_{city}").fetchone()[0]
            logger.info(f"Created fact_listings_daily_{city} with {row_count} rows")
            self.summary["fact"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to build fact_listings_daily for {city}: {e}")
            self.summary["failed"] += 1
            return False

    def load_to_supabase(self, city: str, table_name: str, duckdb_table: str) -> bool:
        """
        Load a DuckDB table to Supabase.

        Uses DuckDB's postgres_scanner for efficient bulk load.
        """
        try:
            conn = self._get_pg_conn()
            cur = conn.cursor()

            logger.info(f"Loading {duckdb_table} to Supabase {table_name}...")

            # Drop table if exists (idempotent)
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

            # Export from DuckDB as Parquet in memory, then bulk insert
            # For simplicity, we'll use DuckDB's postgres_scanner extension
            pg_dsn = (
                f"postgres://{self.settings['supabase']['user']}"
                f":{self.settings['supabase']['password']}"
                f"@{self.settings['supabase']['host']}:"
                f"{self.settings['supabase']['port']}/"
                f"{self.settings['supabase']['database']}"
            )

            # Attach Postgres database in DuckDB
            self.duck.execute(f"ATTACH 'dbname=postgres user=postgres' AS postgres_db")

            # Insert data
            insert_query = f"""
            INSERT INTO postgres_db.public.{table_name}
            SELECT * FROM {duckdb_table}
            """

            self.duck.execute(insert_query)
            logger.info(f"Loaded {duckdb_table} to {table_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            return False

    def run(self, cities: List[str]) -> Dict[str, int]:
        """
        Execute full Gold layer build and load.

        Args:
            cities: List of city names

        Returns:
            Summary dict
        """
        try:
            # Build calendar dimension (shared across all cities)
            self.build_dim_calendar()

            # Build dimensions and facts per city
            for city in cities:
                logger.info(f"Processing {city}...")
                self.build_dim_listings(city)
                self.build_dim_hosts(city)
                self.build_dim_neighbourhoods(city)
                self.build_fact_listings_daily(city)

            logger.info(f"Gold layer build complete. Summary: {self.summary}")
            return self.summary

        finally:
            self.close()

    def close(self):
        """Close connections."""
        self.duck.close()
        if self.pg_conn:
            self.pg_conn.close()


def run(settings_path: str = "./config/settings.yaml", cities: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Entry point for Gold layer build.

    Args:
        settings_path: Path to settings.yaml
        cities: List of cities to process

    Returns:
        Summary dict
    """
    if cities is None:
        cities = ["london", "paris"]

    builder = GoldLayerBuilder(settings_path)
    return builder.run(cities)


if __name__ == "__main__":
    import sys
    from utils import setup_logging

    setup_logging(Path("./logs"), level="INFO", format_type="json")
    result = run()
    sys.exit(0 if result["failed"] == 0 else 1)
