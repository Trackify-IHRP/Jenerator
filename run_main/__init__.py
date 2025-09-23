import azure.functions as func
import logging
from . import main as run_main_main

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing run_main")

    try:
        # Expect ?issue_key=POI-123 or JSON body {"issue_key": "POI-123"}
        issue_key = req.params.get("issue_key")
        if not issue_key:
            try:
                issue_key = req.get_json().get("issue_key")
            except:
                pass

        if not issue_key:
            return func.HttpResponse("Missing issue_key", status_code=400)

        result = run_main_main(issue_key)
        return func.HttpResponse(result)
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {e}", status_code=500)
