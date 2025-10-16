variable "stream_name" {
    type = string
    description = "value"
  
}

variable "shard_count" {
    type = number
    description = "value"
  
}

variable "retention_period" {
    type = number
    description = "value"
  
}

variable "shard_level_metrics" {
    type = list(string)
    description = "value"
    default = [
    "IncomingBytes",
    "IncomingRecords",
    "OutgoingBytes",
    "OutgoingRecords",
    "ReadProvisionedThroughputExceeded",
    "WriteProvisionedThroughputExceeded",
    "IteratorAgeMilliseconds"
  ]
  
}

variable "tags" {
    description = "value"
    default = "mlops-zoomcamp"
  
}