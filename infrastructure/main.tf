terraform {
  required_version = ">=1.0"
  backend "s3" {
    bucket = "akt-tf-state-mlops-zoomcamp"
    key = "mlops-zoomcamp-stg.tfstate"
    region = "us-east-1"
    encrypt = true
    
  }
}

provider "aws" {
    region = var.aws_region
  
}

data "aws_caller_identity" "current_identity" {

}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
}


# rainfall-events
module "source_kinesis_stream" {
    source = "./modules/kinesis"
    stream_name = "${var.source_stream_name}-${var.project_id}"
    retention_period = 48
    shard_count = 2
    tags = var.project_id
  
}


module "output_kinesis_stream" {
    source = "./modules/kinesis"
    stream_name = "${var.output_stream_name}-${var.project_id}"
    retention_period = 48
    shard_count = 2
    tags = var.project_id
  
}


module "s3_bucket" {
  source = "./modules/s3"
  bucket_name = "${var.model_bucket}-${var.project_id}"
  
}

module "ecr_image" {
    source = "./modules/ecr"
    ecr_repo_name = "${var.ecr_repo_name}_${var.project_id}"
    account_id =  local.account_id
    lambda_function_local_path = var.lambda_function_local_path
    docker_image_local_path = var.docker_image_local_path
  
}
