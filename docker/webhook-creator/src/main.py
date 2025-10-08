import os
from mlflow import MlflowClient

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
receiver_uri = os.getenv("MLFLOW_WEBHOOK_RECEIVER_URI")
client = MlflowClient(tracking_uri=tracking_uri)

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
        url=receiver_uri,
        events=[
            "model_version.created",
            "model_version_alias.created"
        ],
        description="Notifies when a model version is created and assigned an alias"
    )
    print(f"Created webhook: {webhook.webhook_id}")
