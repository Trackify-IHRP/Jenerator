# Orchestrator/__init__.py
import azure.durable_functions as df

def orchestrator_function(context: df.DurableOrchestrationContext):
    # Input from the starter
    issue_key = context.get_input()

    # Call activities in sequence; capture results with `yield`
    yield context.call_activity("run_main", issue_key)
    yield context.call_activity("run_train", "go")
    rows = yield context.call_activity("run_extract", "go")

    # Orchestrator must return JSON-serializable data...
    return rows

main = df.Orchestrator.create(orchestrator_function)
