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


# ride-events
module "source_kinesis_stream" {
    source = "./modules/kinesis"
    stream_name = "${var.source_stream_name}-${var.project_id}"
    retention_period = 48
    shard_count = 2
    tags = var.project_id
  
}