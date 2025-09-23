import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        "Hello from run_hello 🎉",
        status_code=200
    )
