# function_app.py
import json
import azure.functions as func

import test_search as TS  # <-- your file

app = func.FunctionApp()

# Health check
@app.route(route="hello", auth_level=func.AuthLevel.ANONYMOUS)
def hello(req: func.HttpRequest) -> func.HttpResponse:
    name = req.params.get("name", "world")
    return func.HttpResponse(f"Hello, {name}!", status_code=200)

# Train once (don’t use the long-running "watch" mode on Azure)
@app.route(route="train", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def train_http(req: func.HttpRequest) -> func.HttpResponse:
    try:
        TS.do_train()
        return func.HttpResponse(
            json.dumps({"ok": True, "message": "Training complete."}),
            mimetype="application/json", status_code=200
        )
    except SystemExit as e:  # e.g., KB file missing
        return func.HttpResponse(str(e), status_code=400)
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)

# Search/query endpoint
@app.route(route="search", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def search_http(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # inputs
        if req.method == "POST":
            body = req.get_json(silent=True) or {}
            query = body.get("query", "")
            top3 = bool(body.get("top3", False))
            as_json = bool(body.get("json", True))
        else:
            query = req.params.get("query", "")
            top3 = req.params.get("top3", "false").lower() == "true"
            as_json = (req.params.get("json", "true").lower() != "false")

        if not query:
            return func.HttpResponse("Missing 'query'.", status_code=400)

        # Build retriever like test_search.cmd_query Export-AzResourceGroup -ResourceGroupName jenerator_group -Resource /subscriptions/48661f75-808e-4247-8abc-a0eb8061f568/resourceGroups/jenerator_group/providers/Microsoft.Web/sites/jenerator

        df = TS.load_kb()
        retriever = TS.HybridRetriever(df)
        idx_path = TS.OUT_DIR / "hybrid.index"
        if hasattr(retriever, "load") and idx_path.exists():
            try:
                retriever.load(str(idx_path))  # type: ignore
            except Exception:
                pass

        hits = retriever.search(query, top_k=5)

        if top3:
            return func.HttpResponse(
                json.dumps(hits[:3], ensure_ascii=False, indent=2),
                mimetype="application/json", status_code=200
            )

        best = TS.pick_best(hits)
        if as_json:
            return func.HttpResponse(
                json.dumps(best, ensure_ascii=False, indent=2),
                mimetype="application/json", status_code=200
            )

        answer = TS.maybe_generate_final_answer(query, best)
        return func.HttpResponse(answer, status_code=200)

    except SystemExit as e:
        return func.HttpResponse(str(e), status_code=400)
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)
