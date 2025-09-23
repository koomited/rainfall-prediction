from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionPipeline
# from mlProject.pipeline.stage_02_data_validation import  DataValidationPipeline
# from mlProject.pipeline.stage_03_data_transformation import DataTransformationPipeline
# from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
# from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
