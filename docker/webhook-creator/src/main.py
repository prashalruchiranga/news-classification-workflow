import os
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RECEIVER_URI = os.getenv("MLFLOW_WEBHOOK_RECEIVER_URI")
MLFLOW_WEBHOOK_SECRET = os.getenv("MLFLOW_WEBHOOK_SECRET")

if not all([TRACKING_URI, RECEIVER_URI, MLFLOW_WEBHOOK_SECRET]):
    raise ValueError("One or more required environment variables are not set")

client = MlflowClient(tracking_uri=TRACKING_URI)

alias_notifier = "multi-event-webhook"
webhooks = client.list_webhooks()
matching_webhooks = [wh for wh in webhooks if wh.name == alias_notifier]

if len(matching_webhooks) > 1:
    raise ValueError(f"Multiple webhooks exists with the name '{alias_notifier}'")
elif len(matching_webhooks) == 1:
    existing = matching_webhooks[0]
    print(f"Webhook already exists: {existing.webhook_id}")
    client.update_webhook(
        webhook_id=existing.webhook_id,
        url=RECEIVER_URI,
        events=[
            "model_version.created",
            "model_version_alias.created"
        ],
        description="Notifies when a model version is created and assigned an alias",
        secret=MLFLOW_WEBHOOK_SECRET
    )
    webhook = client.get_webhook(existing.webhook_id)
    print(f"Updated webhook receiver: {webhook.webhook_id}")
else:
    webhook = client.create_webhook(
        name=alias_notifier,
        url=RECEIVER_URI,
        events=[
            "model_version.created",
            "model_version_alias.created"
        ],
        description="Notifies when a model version is created and assigned an alias",
        secret=MLFLOW_WEBHOOK_SECRET
    )
    print(f"Created webhook: {webhook.webhook_id}")
