"""Airbnb Market Intelligence Pipeline

End-to-end data engineering pipeline for Inside Airbnb market analysis.
Ingests data from Inside Airbnb public servers, transforms through
Medallion lakehouse (Bronze → Silver → Gold), and serves dimensional
analytics via Supabase + Power BI.
"""

__version__ = "1.0.0"
__author__ = "Data Engineering Team"
