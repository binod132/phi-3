from google.cloud import aiplatform
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/adhikaribinod132/phi-3/vertexa-icebrkr.json"
# Initialize the Vertex AI client
aiplatform.init(
    project="brave-smile-424210-m0",  # The project where Vertex AI will run
    location="us-west1",  # The location of Vertex AI resources
    staging_bucket= "gs://testbucketbinod"
)
service_account="vertex-ai@brave-smile-424210-m0.iam.gserviceaccount.com"
# Create and submit a Custom Training Job
job = aiplatform.CustomJob.from_local_script(
    display_name="phi-3-mini-4k-training",
    script_path="train.py",  # The script you already embedded in the Docker image
    container_uri="asia-docker.pkg.dev/brave-smile-424210-m0/mlops-test/phi-3:main",  # Your Docker image
    replica_count=1,  # Number of machines to run the job on
    machine_type="n1-standard-1",  # The machine type
    accelerator_type="NVIDIA_TESLA_T4",  # GPU type
    accelerator_count=1,  # Number of GPUs
)

# Run the training job
job.run()
