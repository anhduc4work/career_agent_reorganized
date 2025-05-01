# tools/__init__.py
# Import các tool từ các module con trong thư mục tools
from .review_cv_tools import review_cv
from .score_jd_tools import score_jobs
from .analyze_market_tools import job_market_analysis
from .recall_memory_tools import recall_memory
from .retrieve_pg_tools import job_search_by_cv, job_search_by_query

# Định nghĩa danh sách các tool sẽ được sử dụng bởi agent
all_tools = [
    job_search_by_cv, job_search_by_query,
    recall_memory,
    review_cv,
    score_jobs, job_market_analysis
]

# (Optional) Export các tên cụ thể nếu muốn import trực tiếp từ 'tools'
__all__ = [
    'review_cv',
    'score_jobs',
    'job_market_analysis',
    'recall_memory',
    'job_search_by_cv',
    'job_search_by_query',
    'all_tools' # Export danh sách tổng hợp
]