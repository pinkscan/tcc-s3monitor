import boto3
from botocore.config import Config
from .config import settings
import io

session = boto3.session.Session()

s3_client = session.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    aws_session_token=settings.AWS_SESSION_TOKEN,
    region_name=settings.AWS_REGION,
    config=Config(signature_version="s3v4")
)

def upload_raw_to_s3(img_id: str, img_bytes: bytes, content_type: str):
    key = f"{settings.RAW_PREFIX}{img_id}.png"

    s3_client.put_object(
        Bucket=settings.S3_BUCKET_RAW,
        Key=key,
        Body=img_bytes,
        ContentType=content_type
    )

    return f"s3://{settings.S3_BUCKET_RAW}/{key}"


def upload_processed_to_s3(img_id: str, pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    key = f"{settings.PROCESSED_PREFIX}{img_id}.png"

    s3_client.put_object(
        Bucket=settings.S3_BUCKET_PROCESSED,
        Key=key,
        Body=buffer.read(),
        ContentType="image/png"
    )

    return f"s3://{settings.S3_BUCKET_PROCESSED}/{key}"


def upload_results_json_to_s3(img_id: str, json_data: dict):
    import json
    data_bytes = json.dumps(json_data).encode("utf-8")

    key = f"{settings.RESULTS_PREFIX}{img_id}.json"

    s3_client.put_object(
        Bucket=settings.S3_BUCKET_PROCESSED,   # results no mesmo bucket da processada
        Key=key,
        Body=data_bytes,
        ContentType="application/json"
    )

    return f"s3://{settings.S3_BUCKET_PROCESSED}/{key}"
