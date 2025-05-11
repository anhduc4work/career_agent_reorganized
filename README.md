Certainly! Here’s a revised README.md for your GitHub repository career_agent_reorganized, designed to clearly present the project’s purpose, features, and setup instructions:

⸻

Career Agent Reorganized

Career Agent Reorganized is an intelligent career assistant system that leverages large language models (LLMs) and modular tools to assist users in job searching, CV evaluation, and career planning. The system integrates various components to provide personalized career guidance.

🚀 Features
	•	CV Review and Enhancement: Analyze and improve CVs based on specific job descriptions.
	•	Job Matching: Recommend job postings that align with the user’s CV and preferences.
	•	Market Analysis: Aggregate and analyze job descriptions to identify skill trends and demand patterns.
	•	Interactive Chat Interface: Engage with the assistant through a conversational UI powered by Gradio.

🛠️ Installation

Prerequisites
	•	Python 3.8 or higher
	•	PostgreSQL
	•	Git

Setup Instructions
	1.	Clone the repository:

git clone https://github.com/anhduc4work/career_agent_reorganized.git
cd career_agent_reorganized


	2.	Install Python dependencies:

pip install -r requirements.txt


	3.	Set up PostgreSQL:
	•	Install PostgreSQL and development libraries:

sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-14


	•	Start PostgreSQL service:

sudo service postgresql start


	•	Create necessary database and tables:

sudo -u postgres psql -c "CREATE TABLE IF NOT EXISTS user_thread (id SERIAL PRIMARY KEY, user_id TEXT, thread_id TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
sudo -u postgres psql -c "ALTER TABLE user_thread ADD CONSTRAINT unique_user_thread UNIQUE (user_id, thread_id);"
sudo -u postgres psql -c "GRANT CONNECT ON DATABASE postgres TO postgres;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON TABLE user_thread TO postgres;"
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'postgres';"


	4.	Install pgvector extension:

cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
sudo service postgresql restart


	5.	Install and configure Ollama:

curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull qwen3:8b



📂 Project Structure
	•	agent/: Core agent logic and workflows.
	•	tools/: Modular tools for CV analysis, job search, and market insights.
	•	interfaces/: Gradio-based user interface components.
	•	data/: Sample data and configurations.
	•	scripts/: Utility scripts for setup and maintenance.

🧪 Testing

The system includes predefined scenarios accessible via the Gradio interface to test various functionalities:
	•	Uploading and analyzing a CV.
	•	Searching for jobs based on keywords or CV content.
	•	Reviewing and enhancing a CV for a specific job description.
	•	Analyzing market trends from multiple job descriptions.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

⸻

Feel free to customize this README.md further to match your project’s specifics. If you need assistance with any particular section or additional features, let me know!