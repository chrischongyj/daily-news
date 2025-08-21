#!/bin/bash
set -a
source .env
set +a

cd /Users/chris/dev/langnews
source .venv/bin/activate
python news_analysis.py --run

#commit
git add .
git commit -m "daily update"
git push

# Automate by adding to crontab
# crontab -e
# @daily /path/to/run_daily.sh