# Start the caplena/c4 datapipeline step

PROJECT=codit-221310
BUCKET_NAME=caplena-c4
MAX_NUM_WORKERS=400
WORKER_IMAGE=eu.gcr.io/codit-221310/cloud-dataflow-apache-beam_python3.7_sdk:2.27.0
RESULT_DIR=5c

python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4/caplena \
  --data_dir=gs://$BUCKET_NAME/$RESULT_DIR \
  --beam_pipeline_options="project=$PROJECT,job_name=c4,flexrs_goal=COST_OPTIMIZED,staging_location=gs://$BUCKET_NAME/binaries,temp_location=gs://$BUCKET_NAME/temp,runner=DataflowRunner,requirements_file=./beam_requirements.txt,experiments=shuffle_mode=service,experiments=use_runner_v2,region=europe-west4,max_num_workers=$MAX_NUM_WORKERS,worker_harness_container_image=$WORKER_IMAGE"