[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CMJbfJPO)
Welcome to the Term Group Project. This repository is structured to ensure
a smooth development and grading experience. Follow these guidelines to avoid
common issues with API keys and large datasets.
1. Repository Structure
Your repository should follow this layout to work correctly with the autograder:

2. Handling Sensitive Information (.env)
Security Alert: Never upload your Anthropic or OpenAI API keys to GitHub.
Store your keys locally in a file named .env .
The provided .gitignore file is pre-configured to ignore .env .
GitHub will automatically block commits containing active API keys for your
protection.
3. Managing Large Data Files
GitHub has a web upload limit of 25MB and a repository limit of 100MB. For this
   Project:
Local Work: Keep your full datasets (e.g., 500MB+ databases) on your local
machine.
├── data/
│ └── sample_data.csv <-- SMALL sample for autograding (max 5MB)
├── main.py <-- Your primary Python logic
├── requirements.txt <-- List of libraries (pandas, numpy, etc.)
├── .gitignore <-- Prevents secret/large file leaks
└── README.md <-- Project documentation


GitHub Upload: Only upload a sample version of your data to the data/
folder. This allows the autograder to verify your code logic without timing out or
failing.

Pro Tip: In your code, use a check to see if the script is running on GitHub or
locally:
if os.getenv('GITHUB_ACTIONS'): data_path = 'data/sample.csv'

4. How to Submit
Commit your code changes locally: git commit -m "update logic" .
Push to the main branch: git push origin main .
Check the Actions tab in GitHub to see if the Autograder passed.
