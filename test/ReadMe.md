# Note to Developers

For testing FoKL as exists in your fork or branch, the process is not yet fully automated. To test,
- manually change the 'CURRENT_REPO_URL' variable in 'tox.ini' to your fork/branch link (e.g., 'https://github.com/your_username/FoKL-GPy/tree/your_branch'); commit
- confirm the tests are successful
- change 'CURRENT_REPO_URL' back to 'https://github.com/ESMS_Group-Public/FoKL-GPy'; commit
- open a pull request to the official repo

To automate, a method using 'get_current_repo_url.py' to define 'CURRENT_REPO_URL' is in progress.
