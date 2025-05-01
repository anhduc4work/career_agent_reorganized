import uuid
import gradio as gr

CV_CONTENT = """ AI Engineer
A highly motivated student with a strong passion for AI, backed by a solid foundation in mathematics,
statistics and programming
anhduc4hl2003@gmail.com   +84911581889     anhduc4work
Experience
AI Researcher Intern  •  AVT Consulting and Technology
Jan 2025 - Present 
Researched and explored pre-trained models for image processing. 
Collaborated with the IS team to integrate AI into the system for the MVP project.
Data Analyst Intern  •  VNPT AI
Jul 2024 - Sep 2024
Designed and implemented real-time dashboards using Power BI.
Achieved 1st place in an internal Data Science competition for interns.
Education
Data Science in Economics University  •  National Economics University
Sep 2021 - May 2025 
GPA: 3.40/4.00
Relevant Courses:
Natural Language Processing 1&2 (A) – Implemented & worked with transformer models, sentiment analysis, and
text classication.
Deep Learning (A) – Implemented neural networks, CNNs, and RNNs for real-world applications.
Big Data (A) – Applied distributed computing frameworks (Spark, Hadoop) for large-scale data processing.
Time Series Analysis (A+) – Forecasting techniques using ARIMA variants.
Physics Major  •  Ha Tinh High School for the Gifted
Sep 2018 - Jun 2021
Certication
IBM: AI Engineering Professional Certicate • Jan 2024
Image processing with OpenCV and Pillow, model building with TensorFlow, Keras, and PyTorch.
IBM: Data Science Professional Certicate • Sep 2023
Mastered Python, SQL, and Machine Learning through hands-on projects.
Others:  Top 10 H4TF Competition: Data Analysis - Unveil Data Power, Top 10FTU: Data Science Talent Competition, Fine
tuning Large Language Models, Building Systems with the ChatGPT API, Machine Learning in Production
(DeepLearning.AI), Data Engineer Professional Certicate, Generative AI for Data Scientists Specialization (IBM),
Introduction to LangGraph (LangChain), ...
Certication
Multi Agent Career Assistant
Developed a multi-agent system to assist job seekers in nding relevant jobs and optimizing their CVs.
Designed a job retrieval pipeline using semantic search with Atlas MongoDB and Nomic Embed Text.
Implemented agent coordination with LangGraph to automate job matching, CV evaluation, and CV adjustment.
Leveraged LLama 3.3 70B to provide personalized feedback and improvements for job applications.
Built a Gradio-basedUI for chat interaction, supporting CV uploads and automatic information extraction.
RAG System on PDF Documents
Developed an AI-powered Retrieval-Augmented Generation (RAG) system for querying and summarizing large PDF
documents.
Implemented document chunking to preprocess and split large PDFs into structured text.
Integrated Llama-based LLMs using LangChain to generate accurate and context-aware summaries.
Optimized retrieval mechanisms to improve response accuracy by 23% compared to LLM-only approach.
Impact of news on VN30 Index
Analyzed nancial news impact on VN30 stock index movements using NLP and ML models.
Collected and labeled 10,000+ nancial news articles from CafeF, VietStock, ...
Engineered textual sentiment scores as features for stock return predictions.
Built predictive models using LightGBM, ... achieving 62% accuracy on historical return data.
Tech stack
Programming Languages: Python, R
Databases: MySQL, PostgreSQL, MongoDB
Frameworks: Pytorch
Languages
Chinese - Basic
English - IELTS"""

def demo_upload_cv_and_search_tool(chat_history):
    chat_history = chat_history + [
            {"role": "user", "content": "Tìm cho tôi 2 công việc phù hợp CV của tôi nhất.", "metadata": {"id": str(uuid.uuid4())}},
            {"role": "user", "content": CV_CONTENT, "metadata": {"title": "File included", "id": str(uuid.uuid4())}},
        ]
    return chat_history

def demo_search_by_query_tool():
    return gr.MultimodalTextbox(value="Tìm cho tôi 3 job data analysts fulltime")

def demo_score_jds_tool(available_job):
    available_job_index = [job['id'] for job in available_job][:2]
    return gr.MultimodalTextbox(value=f"Chấm điểm độ phù hợp của công việc {available_job_index} với cv của tôi.")

def demo_review_cv_tool(available_job):
    available_job_index = [job['id'] for job in available_job][0] or 7363
    return gr.MultimodalTextbox(value=f"Hãy review, chỉnh sửa job {available_job_index} với cv của tôi.")
    
def demo_analyze_market_tool():
    return gr.MultimodalTextbox(value=f"Phân tích thị trường ngành marketing.")  