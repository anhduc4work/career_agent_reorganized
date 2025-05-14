# tools/__init__.py
# Import các tool từ các module con trong thư mục tools
from .match_cv_jd_tools import match_cv_jd
from .score_jd_tools import score_jobs
from .analyze_market_tools import job_market_analysis
from .recall_memory_tools import recall_history_chat, recall_state
from .retrieve_pg_tools import job_search_by_cv, job_search_by_query
from .review_general_cv_tool import review_cv
# Định nghĩa danh sách các tool sẽ được sử dụng bởi agent
all_tools = [
    job_search_by_cv, job_search_by_query,
    recall_history_chat, recall_state,
    review_cv,
    match_cv_jd,
    score_jobs, job_market_analysis
]

# (Optional) Export các tên cụ thể nếu muốn import trực tiếp từ 'tools'
__all__ = [
    'review_cv',
    'score_jobs',
    'job_market_analysis',
    'recall_history_chat', 'recall_state',
    'job_search_by_cv',
    'job_search_by_query',
    "match_cv_jd",
    'all_tools' # Export danh sách tổng hợp
]