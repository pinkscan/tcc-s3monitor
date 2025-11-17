import os

class Settings:
    # Credenciais AWS
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    # Buckets
    S3_BUCKET_RAW = os.getenv("S3_BUCKET_RAW")
    S3_BUCKET_PROCESSED = os.getenv("S3_BUCKET_PROCESSED")

    # Prefixos
    RAW_PREFIX = os.getenv("RAW_PREFIX", "raw/")
    PROCESSED_PREFIX = os.getenv("PROCESSED_PREFIX", "processed/")
    RESULTS_PREFIX = os.getenv("RESULTS_PREFIX", "results/")

settings = Settings()
