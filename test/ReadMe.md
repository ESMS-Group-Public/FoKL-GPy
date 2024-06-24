# Note to Developers

For testing the FoKL-GPy package as exists in the main branch of a developer's fork. The process is not yet fully automated. To test,
- manually change the value of the 'CURRENT_REPO_URL' variable in 'tox.ini' to your fork link (e.g., 'https://github.com/YOUR_USERNAME/FoKL-GPy')
- commit/push all pending changes to the main branch of your fork if not yet done
- the tests will automatically run in your main branch
- debug the changes in your fork until all tests are successful
- change 'CURRENT_REPO_URL' back to 'https://github.com/ESMS_Group-Public/FoKL-GPy' and commit to your main branch
- open a pull request to the official ESMS repo

To automate, a method using 'get_current_repo_url.py' to define 'CURRENT_REPO_URL' is in progress.
