import azure.functions as func
import json
import logging
from . import main as run_extract_main

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing run_extract")

    try:
        result = run_extract_main("_")   # no input needed
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {e}", status_code=500)
