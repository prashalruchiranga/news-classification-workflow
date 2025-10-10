import hmac
import hashlib
import base64
import time

def verify_timestamp_freshness(
    timestamp_str: str, max_age: int
) -> bool:
    """Verify that the webhook timestamp is recent enough to prevent replay attacks"""
    try:
        webhook_timestamp = int(timestamp_str)
        current_timestamp = int(time.time())
        age = current_timestamp - webhook_timestamp
        return 0 <= age <= max_age
    except (ValueError, TypeError):
        return False

def verify_mlflow_signature(
    payload: str, signature: str, secret: str, delivery_id: str, timestamp: str
) -> bool:
    """Verify the HMAC signature from MLflow webhook"""
    # Extract the base64 signature part (remove 'v1,' prefix)
    if not signature.startswith("v1,"):
        return False

    signature_b64 = signature.removeprefix("v1,")
    # Reconstruct the signed content: delivery_id.timestamp.payload
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    # Generate expected signature
    expected_signature = hmac.new(
        secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature_b64 = base64.b64encode(expected_signature).decode("utf-8")
    return hmac.compare_digest(signature_b64, expected_signature_b64)
