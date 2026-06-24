"""
Streaming file ingestion from Inside Airbnb to S3 Bronze layer.

Downloads files, computes hashes, maintains manifest index, handles retries,
and ensures idempotency via hash-based deduplication.
"""

import logging
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from io import BytesIO

import requests
import yaml

from utils import S3Client, compute_streaming_sha256, retry_on_transient

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """Single manifest record for an ingested file."""

    source_url: str
    captured_at: str
    bytes: int
    sha256: str
    rows_est: int
    status: str  # NEW | DUPLICATE | RETRIED
    city: str
    year: int
    month: int


class IngestionEngine:
    """Orchestrates streaming download and Bronze ingestion."""

    def __init__(
        self,
        config_path: Path,
        settings_path: Path,
        dry_run: bool = False,
    ):
        """
        Initialize the ingestion engine.

        Args:
            config_path: Path to cities.yaml
            settings_path: Path to settings.yaml
            dry_run: If True, simulate without writing to S3
        """
        self.config_path = Path(config_path)
        self.settings_path = Path(settings_path)
        self.dry_run = dry_run

        # Load YAML configs
        with open(self.config_path) as f:
            self.cities_config = yaml.safe_load(f)

        with open(self.settings_path) as f:
            self.settings = yaml.safe_load(f)

        # Initialize S3 client
        self.s3 = S3Client(
            region=self.settings["aws"]["region"],
            profile=self.settings["aws"]["profile"],
        )

        self.manifest_by_city: Dict[str, List[ManifestEntry]] = {}
        self.summary = {"downloaded": 0, "skipped": 0, "failed": 0, "retried": 0}

    def _resolve_city_config(self, city_name: str) -> Optional[Dict]:
        """Resolve city config by name."""
        for city in self.cities_config.get("cities", []):
            if city["name"].lower() == city_name.lower():
                return city
        return None

    def _get_manifest_index(self, bucket: str, city: str) -> Dict[str, ManifestEntry]:
        """
        Load manifest index from S3 (if exists).

        Returns a dict: {sha256 -> ManifestEntry} for fast lookup.
        """
        prefix = f"{self.settings['s3']['bronze_prefix']}city={city}/"
        objects = self.s3.list_objects(bucket, prefix, max_keys=10000)

        manifest_index = {}
        for obj in objects:
            if obj["Key"].endswith("_manifest.json"):
                content = self.s3.get_object(bucket, obj["Key"])
                if content:
                    try:
                        manifest_data = json.loads(content.decode())
                        for entry in manifest_data.get("entries", []):
                            manifest_index[entry["sha256"]] = entry
                    except Exception as e:
                        logger.warning(f"Failed to parse manifest {obj['Key']}: {e}")
        return manifest_index

    def _write_manifest(
        self, bucket: str, city: str, year: int, month: int, entries: List[Dict]
    ) -> bool:
        """Write manifest JSON to S3."""
        key = (
            f"{self.settings['s3']['bronze_prefix']}"
            f"city={city}/year={year}/month={month}/_manifest.json"
        )
        manifest = {"entries": entries, "generated_at": datetime.now(timezone.utc).isoformat()}

        try:
            self.s3.put_object(bucket, key, json.dumps(manifest, indent=2).encode())
            logger.info(f"Wrote manifest to {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to write manifest {key}: {e}")
            return False

    @retry_on_transient(max_retries=4, base_delay=1.0)
    def _download_file(self, url: str, timeout: Tuple[int, int] = (10, 60)) -> Tuple[BytesIO, str, int]:
        """
        Stream-download a file and compute SHA256 on-the-fly.

        Returns:
            (BytesIO buffer, sha256 hex, total bytes)
        """
        buffer = BytesIO()
        sha256 = hashlib.sha256()
        total_bytes = 0

        resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        for chunk in resp.iter_content(chunk_size=self.settings["ingestion"]["chunk_size_bytes"]):
            if chunk:
                buffer.write(chunk)
                sha256.update(chunk)
                total_bytes += len(chunk)

        buffer.seek(0)
        return buffer, sha256.hexdigest(), total_bytes

    def _upload_to_bronze(
        self,
        bucket: str,
        buffer: BytesIO,
        city: str,
        year: int,
        month: int,
        filename: str,
    ) -> bool:
        """Upload downloaded file to Bronze."""
        key = (
            f"{self.settings['s3']['bronze_prefix']}"
            f"city={city}/year={year}/month={month}/{filename}"
        )

        buffer.seek(0)
        return self.s3.put_object(
            bucket,
            key,
            buffer.read(),
            metadata={"city": city, "captured_at": datetime.now(timezone.utc).isoformat()},
        )

    def ingest_city(self, city_name: str) -> bool:
        """
        Ingest all files for a city.

        Args:
            city_name: City identifier from cities.yaml

        Returns:
            True if at least one file was ingested successfully
        """
        city_config = self._resolve_city_config(city_name)
        if not city_config:
            logger.error(f"City '{city_name}' not found in config")
            return False

        logger.info(f"Starting ingestion for {city_name}...")

        bucket = self.settings["s3"]["bronze_bucket"]
        year = datetime.now().year
        month = datetime.now().month

        # Load existing manifest
        manifest_index = self._get_manifest_index(bucket, city_name)

        manifest_entries = []
        city_summary = {"downloaded": 0, "skipped": 0, "failed": 0, "retried": 0}

        # Ingest all 7 files per city according to the links in cities.yaml
        file_specs = [
            ("listings.csv.gz", city_config["listings_gz"]),
            ("listings.csv", city_config["listings_csv"]),
            ("calendar.csv.gz", city_config["calendar_gz"]),
            ("reviews.csv.gz", city_config["reviews_gz"]),
            ("reviews.csv", city_config["reviews_csv"]),
            ("neighbourhoods.csv", city_config["neighbourhoods_csv"]),
            ("neighbourhoods.geojson", city_config["neighbourhoods_geojson"]),
        ]

        for filename, url in file_specs:
            try:
                logger.info(f"Downloading {filename} from {url}...")

                buffer, sha256, total_bytes = self._download_file(url)

                # Check for duplicate
                if sha256 in manifest_index:
                    logger.info(f"{filename} already ingested (hash {sha256[:8]}...). Skipping.")
                    city_summary["skipped"] += 1
                    continue

                # Upload to Bronze
                if not self.dry_run:
                    if self._upload_to_bronze(bucket, buffer, city_name, year, month, filename):
                        logger.info(f"Uploaded {filename} ({total_bytes} bytes)")
                        city_summary["downloaded"] += 1
                        manifest_entries.append({
                            "source_url": url,
                            "captured_at": datetime.now(timezone.utc).isoformat(),
                            "bytes": total_bytes,
                            "sha256": sha256,
                            "rows_est": 0,  # Will be computed in transform
                            "status": "NEW",
                        })
                    else:
                        city_summary["failed"] += 1
                else:
                    logger.info(f"[DRY RUN] Would upload {filename} ({total_bytes} bytes)")
                    city_summary["downloaded"] += 1

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code in self.settings["ingestion"]["no_retry_status_codes"]:
                    logger.error(f"{filename}: {status_code} (permanent). Not retrying.")
                else:
                    logger.error(f"{filename}: HTTP {status_code} (transient)")
                city_summary["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to ingest {filename}: {e}")
                city_summary["failed"] += 1

        # Write manifest
        if manifest_entries and not self.dry_run:
            self._write_manifest(bucket, city_name, year, month, manifest_entries)

        logger.info(f"City {city_name} summary: {city_summary}")
        for key, val in city_summary.items():
            self.summary[key] += val

        return city_summary["downloaded"] > 0

    def run(self) -> Dict[str, int]:
        """
        Execute full ingestion pipeline.

        Returns:
            Summary dict {downloaded, skipped, failed, retried}
        """
        enabled_cities = self.cities_config.get("enabled_cities", [])

        if not enabled_cities:
            logger.warning("No enabled cities in config")
            return self.summary

        for city in enabled_cities:
            try:
                self.ingest_city(city)
            except Exception as e:
                logger.error(f"Fatal error ingesting {city}: {e}", exc_info=True)

        logger.info(f"Ingestion complete. Summary: {self.summary}")
        return self.summary


def run(config_path: str = "./config/cities.yaml", settings_path: str = "./config/settings.yaml", dry_run: bool = False) -> Dict[str, int]:
    """
    Entry point for ingestion.

    Args:
        config_path: Path to cities.yaml
        settings_path: Path to settings.yaml
        dry_run: If True, simulate without writing

    Returns:
        Summary dict
    """
    engine = IngestionEngine(config_path, settings_path, dry_run=dry_run)
    return engine.run()


if __name__ == "__main__":
    import sys
    from utils import setup_logging

    setup_logging(Path("./logs"), level="INFO", format_type="json")
    result = run(dry_run=False)
    sys.exit(0 if result["failed"] == 0 else 1)

    
