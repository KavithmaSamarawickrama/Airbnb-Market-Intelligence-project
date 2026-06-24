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
                with self.pg_conn.cursor() as cur:
                    cur.execute("SET statement_timeout = 0;")
                logger.info(f"Connected to Supabase at {sb['host']} and disabled statement timeout")
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
                isodow(full_date) as day_of_week,
                (isodow(full_date) IN (6, 7)) as is_weekend,
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
            city_clean = city.replace('-', '_')

            # Read from Silver Parquet
            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_listings_{city_clean} AS
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
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_listings_{city_clean}").fetchone()[0]
            logger.info(f"Created dim_listings_{city_clean} with {row_count} rows")
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
            city_clean = city.replace('-', '_')

            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_hosts_{city_clean} AS
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
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_hosts_{city_clean}").fetchone()[0]
            logger.info(f"Created dim_hosts_{city_clean} with {row_count} rows")
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
            city_clean = city.replace('-', '_')

            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"
            if not listings_path.exists():
                logger.warning(f"Silver file not found: {listings_path}")
                return False

            query = f"""
            CREATE OR REPLACE TABLE dim_neighbourhoods_{city_clean} AS
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
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM dim_neighbourhoods_{city_clean}").fetchone()[0]
            logger.info(f"Created dim_neighbourhoods_{city_clean} with {row_count} rows")
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
            city_clean = city.replace('-', '_')

            calendar_path = Path("./data/processed_silver") / f"{city}_calendar_clean.parquet"
            listings_path = Path("./data/processed_silver") / f"{city}_listings_clean.parquet"

            for path in [calendar_path, listings_path]:
                if not path.exists():
                    logger.warning(f"Silver file not found: {path}")
                    return False

            # Complex join with dimension tables
            query = f"""
            CREATE OR REPLACE TABLE fact_listings_daily_{city_clean} AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY dl.listing_key, dc.date_key) as fact_id,
                dl.listing_key,
                dh.host_key,
                dn.neighbourhood_key,
                dc.date_key,
                COALESCE(cal.price, lst.price) as price,
                cal.is_available,
                cal.minimum_nights,
                COALESCE(COALESCE(cal.price, lst.price) * (CASE WHEN cal.is_available THEN 1.0 ELSE 0.0 END), 0) as est_revenue,
                COALESCE(lr.review_count, 0) as reviews_to_date
            FROM read_parquet('{calendar_path}') as cal
            LEFT JOIN read_parquet('{listings_path}') as lst
                ON cal.listing_id = lst.listing_id
            LEFT JOIN dim_listings_{city_clean} dl
                ON lst.listing_id = dl.listing_id
            LEFT JOIN dim_hosts_{city_clean} dh
                ON lst.host_id = dh.host_id
            LEFT JOIN dim_neighbourhoods_{city_clean} dn
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
            row_count = self.duck.execute(f"SELECT COUNT(*) FROM fact_listings_daily_{city_clean}").fetchone()[0]
            logger.info(f"Created fact_listings_daily_{city_clean} with {row_count} rows")
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
            conn.commit()

            # Export from DuckDB as Parquet in memory, then bulk insert
            # For simplicity, we'll use DuckDB's postgres_scanner extension
            pg_dsn = (
                f"postgres://{self.settings['supabase']['user']}"
                f":{self.settings['supabase']['password']}"
                f"@{self.settings['supabase']['host']}:"
                f"{self.settings['supabase']['port']}/"
                f"{self.settings['supabase']['database']}"
                f"?options=-c%20statement_timeout%3D0"
            )

            # Detach if already attached from a previous run or city
            try:
                self.duck.execute("DETACH postgres_db")
            except Exception:
                pass

            # Attach Postgres database in DuckDB using the DSN
            self.duck.execute(f"ATTACH '{pg_dsn}' AS postgres_db (TYPE POSTGRES)")

            # Create and Insert data
            insert_query = f"""
            CREATE TABLE postgres_db.public.{table_name} AS
            SELECT * FROM {duckdb_table}
            """

            self.duck.execute(insert_query)
            logger.info(f"Loaded {duckdb_table} to {table_name}")

            # Detach after loading to clean up connection
            try:
                self.duck.execute("DETACH postgres_db")
            except Exception:
                pass

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
            city_cleans = []
            for city in cities:
                logger.info(f"Processing {city}...")
                self.build_dim_listings(city)
                self.build_dim_hosts(city)
                self.build_dim_neighbourhoods(city)
                self.build_fact_listings_daily(city)
                city_cleans.append(city.replace('-', '_'))

            if city_cleans:
                logger.info("Unioning and conforming multi-city Gold layers...")
                
                # 1. Conformed Listings Dimension
                union_listings = " UNION ALL ".join([f"SELECT * FROM dim_listings_{c}" for c in city_cleans])
                self.duck.execute(f"""
                    CREATE OR REPLACE TABLE dim_listings AS
                    SELECT 
                        ROW_NUMBER() OVER (ORDER BY listing_id) as listing_key,
                        listing_id, name, room_type, property_type, accommodates, bedrooms, beds, latitude, longitude
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY listing_id ORDER BY listing_key) as rn
                        FROM ({union_listings})
                    ) WHERE rn = 1
                """)
                
                # 2. Conformed Hosts Dimension
                union_hosts = " UNION ALL ".join([f"SELECT * FROM dim_hosts_{c}" for c in city_cleans])
                self.duck.execute(f"""
                    CREATE OR REPLACE TABLE dim_hosts AS
                    SELECT 
                        ROW_NUMBER() OVER (ORDER BY host_id) as host_key,
                        host_id, host_name, is_superhost, host_since, host_listings_count
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY host_id ORDER BY host_key) as rn
                        FROM ({union_hosts})
                    ) WHERE rn = 1
                """)
                
                # 3. Conformed Neighbourhoods Dimension
                union_neighs = " UNION ALL ".join([f"SELECT * FROM dim_neighbourhoods_{c}" for c in city_cleans])
                self.duck.execute(f"""
                    CREATE OR REPLACE TABLE dim_neighbourhoods AS
                    SELECT 
                        ROW_NUMBER() OVER (ORDER BY neighbourhood) as neighbourhood_key,
                        neighbourhood, neighbourhood_group, centroid_lat, centroid_lon
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY neighbourhood, neighbourhood_group ORDER BY neighbourhood_key) as rn
                        FROM ({union_neighs})
                    ) WHERE rn = 1
                """)

                # 4. Conformed Fact Table (mapping back to conformed dimensions)
                fact_queries = []
                for c in city_cleans:
                    fact_queries.append(f"""
                        SELECT
                            f.fact_id,
                            new_dl.listing_key,
                            new_dh.host_key,
                            new_dn.neighbourhood_key,
                            f.date_key,
                            f.price,
                            f.is_available,
                            f.minimum_nights,
                            f.est_revenue,
                            f.reviews_to_date
                        FROM fact_listings_daily_{c} f
                        JOIN dim_listings_{c} old_dl ON f.listing_key = old_dl.listing_key
                        JOIN dim_listings new_dl ON old_dl.listing_id = new_dl.listing_id
                        
                        JOIN dim_hosts_{c} old_dh ON f.host_key = old_dh.host_key
                        JOIN dim_hosts new_dh ON old_dh.host_id = new_dh.host_id
                        
                        JOIN dim_neighbourhoods_{c} old_dn ON f.neighbourhood_key = old_dn.neighbourhood_key
                        JOIN dim_neighbourhoods new_dn 
                            ON old_dn.neighbourhood = new_dn.neighbourhood 
                            AND COALESCE(old_dn.neighbourhood_group, '') = COALESCE(new_dn.neighbourhood_group, '')
                    """)
                
                union_facts = " UNION ALL ".join(fact_queries)
                self.duck.execute(f"""
                    CREATE OR REPLACE TABLE fact_listings_daily_snapshot AS
                    SELECT 
                        ROW_NUMBER() OVER () as fact_id,
                        listing_key, host_key, neighbourhood_key, date_key,
                        price, is_available, minimum_nights, est_revenue, reviews_to_date
                    FROM ({union_facts})
                """)

                # Upload conformed tables to Supabase
                logger.info("Uploading conformed tables to Supabase...")
                self.load_to_supabase("all", "dim_calendar_dates", "dim_calendar_dates")
                self.load_to_supabase("all", "dim_listings", "dim_listings")
                self.load_to_supabase("all", "dim_hosts", "dim_hosts")
                self.load_to_supabase("all", "dim_neighbourhoods", "dim_neighbourhoods")
                self.load_to_supabase("all", "fact_listings_daily_snapshot", "fact_listings_daily_snapshot")
                
                # Run analytics view scripts to create views in Supabase
                try:
                    conn = self._get_pg_conn()
                    cur = conn.cursor()
                    with open("sql/30_analytics.sql") as f:
                        views_sql = f.read()
                    cur.execute(views_sql)
                    conn.commit()
                    logger.info("Successfully created analytics views on Supabase")
                except Exception as e:
                    logger.error(f"Failed to create analytics views on Supabase: {e}")

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
        try:
            config_path = Path(settings_path).parent / "cities.yaml"
            with open(config_path) as f:
                cities_config = yaml.safe_load(f)
            cities = cities_config.get("enabled_cities", [])
        except Exception as e:
            logger.warning(f"Could not load cities.yaml, defaulting to london, paris: {e}")
            cities = ["london", "paris"]

    builder = GoldLayerBuilder(settings_path)
    return builder.run(cities)


if __name__ == "__main__":
    import sys
    from utils import setup_logging

    setup_logging(Path("./logs"), level="INFO", format_type="json")
    result = run()
    sys.exit(0 if result["failed"] == 0 else 1)
