# # --- INPUT FILES ---
# EXCEL_PATH = "issue_comments_table"     # change this to your file

# # SHEET NAMES (or None to auto-detect first sheet)
# ISSUES_SHEET   = "POI Issues"
# COMMENTS_SHEET = "POI Comments"

# # COLUMN MAPS (edit to match your Excel headers)
# ISSUE_COLS = {
#     "issue_id": "Key",                 # e.g. "POI-1"
#     "problem_text": "Problem/Summary", # short description
#     "created_at": "Created",           # optional
#     "tags": "Module"                   # optional
# }

# COMMENT_COLS = {
#     "issue_id": "POI Ticket",     # joins to Issues.Key
#     "comment_id": "Comment ID",
#     "text": "Comment Text",       # plain text (not JSON)
#     "author_role": "Author Role", # "staff" / "user" (optional)
#     "created_at": "Created"       # date/time (optional)
# }

# # LLM SETTINGS (fill if you want auto-calls; otherwise leave blank)
# LLM_PROVIDER = "openai"  # "openai" | "azure" | "none"
# OPENAI_MODEL = "gpt-4o-mini"  # pick your model
# MAX_COMMENTS = 40   # cap comments per ticket in prompt






# --- INPUT FILES ---
# Use your exact Excel path (supports .xlsx/.xlsm)
# EXCEL_PATH = r"C:\Users\JumanaHASEEN\Institute for Human Resource Professionals LTD\IHRP-TRACKIFY\Trackify\issue_table.xlsm"
EXCEL_PATH = r"C:\Users\JumanaHASEEN\Institute for Human Resource Professionals LTD\IHRP - Digital and Technology - 36. Trackify\issue_table.xlsm"


# SHEET NAMES (must match your Excel tabs exactly)
ISSUES_SHEET   = "Table 1"
COMMENTS_SHEET = "Table 2"  

# COLUMN MAPS (match your headers)
# For "POI Issues"
ISSUE_COLS = {
    "issue_id": "Key",        # e.g. "POI-1"
    "problem_text": "Status" # short problem line
    # Add only if they exist:
    # "created_at": "Created",
    # "tags": "Module",
}

# For "POI Comments"
COMMENT_COLS = {
    "issue_id": "POI Ticket",  # joins to Issues.Key
    "comment_id": "JiraCommentID",
    "text": "Comments",         # your cleaned plain-text comment column
    # Add only if present:
    # "author_role": "Author Role",
    # "created_at": "Created",
}

# LLM SETTINGS (leave as "none" to skip calling any model)
LLM_PROVIDER = "openai"          
OPENAI_MODEL = "gpt-4o-mini"

# Limit how many comments per ticket we include in the prompt
MAX_COMMENTS = 40

#added
# === SEARCH / GENERATION UPGRADES ===
# Turn on hybrid retrieval (TF-IDF + optional BM25 + optional embeddings)
USE_BM25 = True
USE_EMBEDDINGS = True

# Embeddings settings (skips automatically if no API key)
EMBED_PROVIDER = "openai"
EMBED_MODEL = "text-embedding-3-small"   # cheap + good
EMBED_BATCH = 256

# Hybrid scoring weights (sum to ~1.0 is nice, but not required)
WEIGHT_TFIDF = 0.55
WEIGHT_BM25  = 0.25
WEIGHT_EMB   = 0.20

# Re-rank the top K candidates with semantic (embedding) similarity
RERANK_TOP_K = 20

# Make the final answer sound like a support agent (LLM optional)
ANSWER_WITH_LLM = True          # If False, we’ll format a clean rule-based answer
OPENAI_MODEL_ANSWER = "gpt-4o-mini"

# Cache files
CACHE_DIR = "out"
EMBED_CACHE_FILE = "kb_embed.parquet"
