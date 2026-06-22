import os
from dotenv import load_dotenv
import psycopg2

# Load the environment variables from the .env file in your root folder
load_dotenv()

def test_supabase_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('SUPABASE_HOST'),
            port=6543,
            database='postgres',
            user='postgres.dbrijqmxmfxyyqrabbln',
            password=os.getenv('SUPABASE_PASSWORD'),
            sslmode='require'
        )
        print('✓ Supabase connection successful')
        conn.close()
    except Exception as e:
        print(f'✗ Supabase connection failed: {e}')

if __name__ == "__main__":
    test_supabase_connection()
