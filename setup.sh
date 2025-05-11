#!/bin/bash
set -e  # Exit immediately if any command fails


cd setup_script

pip install gdown
echo "==> Running data setup Python script..."
python3 setup_1_data.py

echo "==> Running service setup Bash script..."

chmod +x setup_2_service.sh
bash setup_2_service.sh

echo "==> Running embedding Python script..."
python3 setup_3_embed.py

echo "==> All steps completed successfully."