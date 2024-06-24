import os
import requests


def get_current_repo_url():
    # Get the GitHub repository owner and name from the environment variables (so that FoKL as existing in this repo is installed by 'requirements.txt')
    github_repository_owner = os.getenv('GITHUB_REPOSITORY_OWNER')
    github_repository_name = os.getenv('GITHUB_REPOSITORY_NAME')

    # Construct the GitHub API URL
    github_api_url = f'https://api.github.com/repos/{github_repository_owner}/{github_repository_name}'

    # Send a GET request to the GitHub API
    response = requests.get(github_api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Extract the repository URL
        repository_url = data['html_url']

        return repository_url
    else:
        raise ValueError(f'Failed to fetch repository URL: {response.status_code}')


# Get url
repo_url = get_current_repo_url()
print(repo_url)