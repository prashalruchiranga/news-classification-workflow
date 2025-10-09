import os
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RECEIVER_URI = os.getenv("MLFLOW_WEBHOOK_RECEIVER_URI")
MLFLOW_WEBHOOK_SECRET = os.getenv("MLFLOW_WEBHOOK_SECRET")

client = MlflowClient(tracking_uri=TRACKING_URI)

alias_notifier = "multi-event-webhook"
webhooks = client.list_webhooks()
for wh in webhooks:
    if wh.name == alias_notifier:
        print(f"Webhook already exists: {wh.webhook_id}")
        webhook = wh
        break
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
