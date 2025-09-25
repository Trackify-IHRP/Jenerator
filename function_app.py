import azure.functions as func

app = func.FunctionApp()

@app.function_name(name="hello")
@app.route(route="hello", auth_level=func.AuthLevel.ANONYMOUS)
def hello(req: func.HttpRequest) -> func.HttpResponse:
    name = req.params.get("name", "world")
    return func.HttpResponse(f"Hello, {name}!", status_code=200)
