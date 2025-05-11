#!/bin/bash

pip install -r requirements.txt

# Install PostgreSQL and dev libraries
apt-get update
apt-get -yq install postgresql postgresql-contrib postgresql-server-dev-14

# Start PostgreSQL
/etc/init.d/postgresql start

# Install pgvector
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # may need sudo depending on environment

# Restart PostgreSQL to load extension
/etc/init.d/postgresql restart

sudo -u postgres psql -c "CREATE TABLE IF NOT EXISTS user_thread (user_id TEXT, thread_id TEXT);"
sudo -u postgres psql -c "ALTER TABLE user_thread ADD CONSTRAINT unique_user_thread UNIQUE (user_id, thread_id);"
sudo -u postgres psql -c "GRANT CONNECT ON DATABASE postgres TO postgres;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON TABLE user_thread TO postgres;"
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'postgres';"


curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull qwen3:8b