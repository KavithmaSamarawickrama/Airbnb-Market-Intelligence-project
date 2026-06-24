# Airbnb Market Intelligence Pipeline

End-to-end data engineering project: ingest Inside Airbnb public datasets, transform through a Medallion lakehouse (Bronze → Silver → Gold), build a dimensional star schema, and serve analytics via Power BI.

## 📐 Architecture

```
Inside Airbnb Web Servers
    ↓ (streaming HTTP)
EC2 t3.medium (Python + DuckDB)
    ↓ (S3 put_object)
S3 Bronze (immutable .csv.gz)
    ↓ (read_csv_auto)
DuckDB (in-process transformation)
    ↓ (COPY TO parquet)
S3 Silver (clean .parquet, partitioned)
    ↓ (build star schema)
DuckDB (dimensions + fact tables)
    ↓ (INSERT via postgres_scanner)
Supabase Gold (managed Postgres, free tier)
    ↓ (DirectQuery)
Power BI (analytics & visualization)
```

## 📊 Medallion Layers

| Layer | Storage | Format | Purpose |
|-------|---------|--------|---------|
| **Bronze** | S3 | Raw .csv.gz | Byte-for-byte immutable source copies; full lineage |
| **Silver** | S3 | Typed .parquet | Cleaned, de-duped, validated; partitioned by city |
| **Gold** | Supabase (Postgres) | Star schema | Query-shaped dimensions + facts; serves BI layer |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- AWS account with S3 access
- Supabase project (free tier: https://supabase.com)
- Docker & Docker Compose (for local Postgres)

### 1. Clone & Install

```bash
git clone <repo>
cd Airbnb-Market-Intelligence-project
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root with Supabase credentials:

```bash
# .env
SUPABASE_HOST=your-project.supabase.co
SUPABASE_PASSWORD=your-anon-key
AWS_PROFILE=default  # or your AWS profile
AWS_REGION=us-east-1
```

Update `config/cities.yaml` to add/remove cities and `config/settings.yaml` for AWS/Supabase details.

### 3. Run Ingestion

```bash
python src/ingest.py
```

Downloads Inside Airbnb CSVs to S3 Bronze, computes hashes for deduplication.

### 4. Run Transformation

```bash
python src/transform.py
```

Cleans and types Bronze CSV files, exports to Silver Parquet.

### 5. Build Gold Layer

```bash
python src/build_gold.py
```

Constructs star schema (dimensions + fact tables), loads to Supabase.

### 6. Profile Data Quality

```bash
python src/profiling.py
```

Generates null rates, cardinality, and outlier detection.

### 7. Connect Power BI

In Power BI Desktop:
1. Get Data → PostgreSQL
2. Enter Supabase host, port 5432, database `postgres`
3. Select views: `v_daily_market_snapshot`, `v_host_performance`, `v_neighbourhood_trends`
4. Load & build visuals

---

## 🏗️ AWS CDK Deployment (Alternative to Manual Setup)

Instead of manually managing EC2 and S3, use **AWS CDK** to deploy the entire infrastructure:

```bash
cd infrastructure/

# 1. Install CDK CLI (one-time)
npm install -g aws-cdk

# 2. Bootstrap AWS account (first-time only)
cdk bootstrap aws://YOUR_ACCOUNT_ID/us-east-1

# 3. Deploy infrastructure
cdk deploy
```

**What CDK Deploys:**
- ✅ EC2 t3.medium instance (auto-stop at 8 PM UTC)
- ✅ S3 Bronze bucket (versioned, encrypted, Glacier transition)
- ✅ S3 Silver bucket (versioned, encrypted)
- ✅ IAM role with scoped permissions (S3 + CloudWatch only)
- ✅ VPC & security group
- ✅ CloudWatch logs
- ✅ EventBridge auto-stop/start schedule (saves 70% on compute costs)

**Cost:** ~$5–10/week with auto-stop schedule (vs $30+/month without)

**See [infrastructure/CDK_DEPLOYMENT_GUIDE.md](infrastructure/CDK_DEPLOYMENT_GUIDE.md) for complete deployment instructions.**

---

## 📁 Project Structure

```
airbnb-market-intel/
├── config/
│   ├── cities.yaml              # Cities to ingest, URLs
│   └── settings.yaml            # AWS, Supabase, DuckDB, logging config
├── src/
│   ├── ingest.py                # Bronze ingestion engine
│   ├── transform.py             # Silver transformation (DuckDB)
│   ├── build_gold.py            # Star schema builder
│   ├── profiling.py             # Data quality profiling
│   └── utils/
│       ├── s3_client.py         # S3 wrapper
│       ├── retry.py             # Exponential backoff
│       ├── hashing.py           # SHA256 streaming
│       ├── logging_config.py    # JSON logging setup
│       └── __init__.py
├── data/
│   ├── raw_bronze/              # Local Bronze mirror (gitignored)
│   └── processed_silver/        # Local Silver mirror (gitignored)
├── sql/
│   ├── 10_dims.sql              # Dimension creation
│   ├── 20_fact.sql              # Fact table creation
│   └── 30_analytics.sql         # Analytics views
├── infrastructure/              # AWS CDK Infrastructure as Code
│   ├── app.py                   # CDK application entry point
│   ├── airbnb_pipeline/
│   │   ├── __init__.py
│   │   └── airbnb_stack.py      # Stack definition (EC2, S3, IAM, VPC)
│   ├── cdk.json                 # CDK configuration
│   ├── requirements.txt         # CDK dependencies
│   ├── CDK_DEPLOYMENT_GUIDE.md  # Complete deployment instructions
│   └── README.md
├── logs/                        # Pipeline logs (gitignored)
├── docker-compose.yml           # Local Postgres + pgAdmin
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔧 Key Features

### Idempotent Ingestion
- Streams downloads (no full RAM load) via `requests.get(stream=True, chunk_size=1MB)`
- Computes SHA256 on-the-fly; skips duplicates via manifest index
- Retries transient failures (5xx, timeout) with exponential backoff; fails fast on permanent errors (404)

### Efficient Transformation
- DuckDB in-process (zero AWS costs for compute)
- Typed Parquet output with Snappy compression
- Partitioning by city for scalability

### Cost-Conscious Architecture
- Single EC2 t3.medium ($0.04/hour, auto-stop)
- S3 storage ~$1–2/month (compressed files)
- Supabase free tier (Postgres) for serving ($0)
- **Target: use <5% of $100 AWS credit in one-week sprint**

### Data Quality
- Null rate profiling per column
- Outlier detection (configurable stddev threshold)
- Cardinality & distribution analysis
- Validation warnings for low-quality datasets

## 📊 Star Schema

**Grain:** One row per listing per calendar date

### Fact Table: `fact_listings_daily_snapshot`
| Column | Type | Purpose |
|--------|------|---------|
| fact_id | BIGINT PK | Unique fact key |
| listing_key | INT FK → dim_listings | Listing surrogate key |
| host_key | INT FK → dim_hosts | Host surrogate key |
| neighbourhood_key | INT FK → dim_neighbourhoods | Neighbourhood surrogate key |
| date_key | INT FK → dim_calendar_dates | Date surrogate key |
| price | NUMERIC | Nightly listing price |
| is_available | BOOLEAN | Available on this date |
| minimum_nights | INT | Min nights to book |
| est_revenue | NUMERIC | price × availability proxy |
| reviews_to_date | INT | Cumulative reviews |

### Dimensions
- **dim_listings**: Room type, property type, amenities, location (lat/lon)
- **dim_hosts**: Superhost status, tenure, listings count
- **dim_neighbourhoods**: Geography, centroid coordinates
- **dim_calendar_dates**: Date attributes (year, month, day_of_week, season, is_weekend)

## 🔐 Security & Best Practices

- S3 buckets private (no public ACL)
- Server-side encryption (AES256) on all S3 objects
- IAM role on EC2 (no explicit AWS credentials in code)
- Structured JSON logging (no sensitive data logged)
- Postgres password via `.env` (never committed)
- Connection pooling to Supabase (5 concurrent)

## 📈 Scalability Path

- **Current:** 3–5 cities, ~1 week sprint
- **Near-term:** 20+ cities; add `year/month` partitioning to S3 keys
- **Future:** Glue for distributed ETL, Redshift for larger datasets

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📚 References

- [Inside Airbnb](http://insideairbnb.com/): Public Airbnb data source
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Supabase Free Tier](https://supabase.com/docs)
- [Kimball Dimensional Modeling](https://www.kimballgroup.com/)

## 📝 License

MIT License (for learning & assessment purposes)

---

**See `.vscode/instructions.md` for step-by-step execution guide.**