import azure.durable_functions as df
 
def orchestrator_function(context: df.DurableOrchestrationContext):
    issue_key = context.get_input()  # e.g., "POI-123"
    yield context.call_activity("run_main", issue_key)
    yield context.call_activity("run_train", "go")  # make conditional later if needed
    rows = yield context.call_activity("run_extract", "go")
    return rows
 
main = df.Orchestrator.create(orchestrator_function)
 import json, azure.functions as func, azure.durable_functions as df
 
async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    client = df.DurableOrchestrationClient(starter)
    body = req.get_json(silent=True) or {}
    issue_key = body.get("issueKey")
    if not issue_key:
        return func.HttpResponse(json.dumps({"error":"Missing issueKey"}), status_code=400, mimetype="application/json")
    instance_id = await client.start_new("orchestrator_function", None, issue_key)
    return client.create_check_status_response(req, instance_id)