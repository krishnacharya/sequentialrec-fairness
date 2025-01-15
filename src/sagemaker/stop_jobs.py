import boto3
import sys

if __name__ == "__main__":
    client = boto3.client('sagemaker') # make sure that the region is right, aws configure
    config_filename = sys.argv[1]
    if config_filename.endswith(".json"):
        jobs = json.load(open(sys.argv[1], "r"))
        # TODO for json
    elif config_filename.endswith(".txt"):
        jobs = []
        for job in open(config_filename):
            jobs.append(job.strip())
            job_status = client.describe_training_job(TrainingJobName = jobs[-1])["TrainingJobStatus"]

            if job_status == 'Stopped':
                print(f"Training job  {jobs[-1]} is already stopped")
            elif job_status == 'Stopping':
                print(f"Training job  {jobs[-1]} stopping in progress")
            elif job_status == 'Failed':
                print(f"Training job  {jobs[-1]} had failed, so no need to stop")
            elif job_status == 'InProgress':
                print(f"Going ahead and stopping the InProgress job {jobs[-1]}")
                client.stop_training_job(TrainingJobName=jobs[-1])
            elif job_status == 'Completed':
                print(f"Job  {jobs[-1]} is already completed")
            else:
                raise AttributeError(f"Unknown job status")
