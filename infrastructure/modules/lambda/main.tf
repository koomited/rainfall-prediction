resource "aws_lambda_function" "kinesis_lambda" {
    function_name = var.lambda_function_name

    image_uri = var.image_uri
    package_type = "Image"
    role = aws_iam_role.iam_lambda.arn
    tracing_config {
      mode = "Active"
    }

    environment {
      variables = {
        PREDICTIONS_STREAM_NAME = var.output_stream_name
        MODEL_BUCKET = var.model_bucket
      }
    }
    timeout = 300
  
}

resource "aws_lambda_function_event_invoke_config" "kinesis_lambda_event" {
  function_name                = aws_lambda_function.kinesis_lambda.function_name
  maximum_event_age_in_seconds = 60
  maximum_retry_attempts       = 0
}

resource "aws_lambda_event_source_mapping" "kinesis_mapping" {
  event_source_arn =  aws_lambda_function.kinesis_lambda.arn
  function_name = aws_lambda_function.kinesis_lambda.arn
  starting_position = "LATEST"
  depends_on = [ 
    aws_iam_policy.allow_kinesis_processing
   ]
}

module "lambda_function" {
    source = "./modules/lambda"
    image_uri = module.ecr_image.image_uri
    lambda_function_name = "${var.lambda_function_name}-${var.project_id}"
    model_bucket = module.s3_bucket.name
    output_stream_name = "${var.output_stream_name}-${var.project_id}"
    output_stream_arn = module.output_kinesis_stream.stream_arn
    source_stream_name = "${var.source_stream_name}-${var.project_id}"
    source_stream_arn = module.source_kinesis_stream.stream_arn
  
}