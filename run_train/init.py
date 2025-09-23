import azure.functions as func
import logging
from . import main as run_train_main

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing run_train")

    try:
        result = run_train_main("_")  # no input needed
        return func.HttpResponse(result)
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {e}", status_code=500)
