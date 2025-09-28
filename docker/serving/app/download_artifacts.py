import os
from google.cloud.storage import Client, transfer_manager

def download_many_blobs_with_transfer_manager(
    bucket_name, blob_names, destination_directory="", workers=8
):
    """Download blobs in a list by name, concurrently in a process pool"""

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

def get_blob_names_with_prefix(bucket_name, prefix):
    """Return all the names of blobs in the bucket that begin with the prefix"""

    # Note: Client.list_blobs requires at least package version 1.17.0.
    # Note: The call returns a response only when the iterator is consumed.
    storage_client = Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]

def main():
    storage_uri = os.getenv("AIP_STORAGE_URI")
    bucket_name, prefix = storage_uri.replace("gs://", "").split("/", 1)
    blob_names = get_blob_names_with_prefix(bucket_name, prefix)
    download_many_blobs_with_transfer_manager(bucket_name, blob_names)

if __name__ == "__main__":
    main()
