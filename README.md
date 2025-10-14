## News Classification MLOps
## Overview
This is an end-to-end news classification workflow implemented using Kubeflow, MLflow and Google Cloud Platform. It automates model training, evaluation, and validation with Kubeflow pipelines. Experiments and model versions are tracked using MLflow with Google Cloud Storage (GCS) as artifact storage and PostgreSQL as persistent backend store. The workflow includes a webhook receiver HTTPS service running on Cloud Run, which is triggered by MLflow events, and deploys trained models to Vertex AI for scalable inference. The deployment uses a custom serving container image running FastAPI, allowing flexible inference with custom dependencies and configurations for the model.

The diagram below illustrates the architectural overview of the project.

<img src="https://github.com/prashalruchiranga/news-classification-mlops/raw/main/images/diagram.png" 
     alt="Architectural overview" width="750">

## Training

The dataset used is **AG's News Topic Classification Dataset** by Xiang Zhang. It is constructed by choosing the four largest classes from the original AG corpus, which is a collection of more than 1 million news articles. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000, and the total number of testing samples is 7,600. You can find the dataset [here](https://huggingface.co/datasets/sh0416/ag_news). The **TinyBERT** model is fine-tuned using this dataset for News Classification. The selected model, `bert_tiny_en_uncased`, is a variant of BERT (Bidirectional Encoder Representations from Transformers), which performs transformer distillation at both the pre-training and task-specific learning stages. It is a 2-layer BERT model with all inputs lowercased, pre-trained on English Wikipedia and the BooksCorpus. The pre-trained base model is available on [kaggle](https://www.kaggle.com/models/keras/bert/keras/bert_tiny_en_uncased/3).

## Technologies Used
- **Kubernetes** - Infrastructure layer for running Kubeflow and MLflow services
- **Kubeflow** - Pipeline orchestration and workflow automation
- **MLflow** - Experiment tracking and model versioning
- **PostgreSQL** - MLflow backend store
- **Google Cloud Storage (GCS)** - MLflow artifact store
- **Docker** - Build container images
- **Google Artifact Registry** - Private registry for storing container images used in model deployment
- **Docker Hub** - Public registry for hosting container images used in the workflow
- **Cloud Run** - Serveless hosting for MLflow webhook receiver
- **Vertex AI** - Model deployment and serving

## Prerequisites
- A Kubernetes cluster
- A Google Cloud Platform (GCP) account

## Getting Started
Below is a minimal setup guide to get started. Please note that this is not an exhaustive installation guide. For detailed instructions, refer to the official Kubeflow, MLflow, and GCP documentation.

1. Deploy Kubeflow pipelines in your cluster. Store `GOOGLE_APPLICATION_CREDENTIALS` as a Kubernetes secret so that Kubeflow components have access to the Artifact Registry.

2. Deploy MLflow in your cluster using a persistent backend (e.g., PostgreSQL) for the metadata store and cloud storage (e.g., GCS) for artifacts. You may need to expose `MLFLOW_TRACKING_URI` to Kubeflow components via ConfigMaps.

3. Configure the webhook secret. :
   - Store the webhook secret as a Kubernetes secret.
   - Generate a secure encryption key for webhook secrets (Refer to the MLflow webhook documentation for detailed instructions)
   - Replace `MLFLOW_TRACKING_URI` and `MLFLOW_WEBHOOK_RECEIVER_URI` in `create-webhook-job.yaml`.
   - Apply the manifest to create the webhook and ensure the job runs to completion.

4. Build and push Docker images to a container registry of your choice. Note that Vertex AI requires custom serving images to be stored in the Google Artifact Registry.

5. Deploy the Webhook Receiver on Google Cloud Run:
    - Set up an HTTPS server that listens on a port of your choice for MLflow model registry events.
    - Provide `MLFLOW_WEBHOOK_SECRET`, `MLFLOW_TRACKING_URI`, `PROJECT_ID`, `LOCATION`, and `CONTAINER_IMAGE_URI` as environment variables. (`PROJECT_ID` and `LOCATION` refer to your GCP project.)
6. Configure GCP account permissions to ensure all necessary services (Artifact Registry, Cloud Run, GCS, etc.) are accessible.

7. Update configuration parameters in `configs/training.yaml` as needed, then proceed with model training.

## Documentation
1. [Deploying Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
2. [GitHub Community Charts - MLflow Chart Usage](https://community-charts.github.io/docs/charts/mlflow/usage)
3. [Setting up MLflow with PostgreSQL as the backend database](https://community-charts.github.io/docs/charts/mlflow/postgresql-backend-installation)
4. [Configuring MLflow to use Google Cloud Storage (GCS) for artifact storage](https://community-charts.github.io/docs/charts/mlflow/google-cloud-storage-integration)
5. [MLflow Webhooks](https://mlflow.org/docs/3.3.2/ml/webhooks/)
6. [Use a custom container for inference in Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container)
7. [Comprehensive documentation, guides, and resources for Google Cloud products and services](https://cloud.google.com/docs)

## License
Licensed under MIT. See the [LICENSE](https://github.com/prashalruchiranga/news-classification-mlops/blob/main/LICENSE).
