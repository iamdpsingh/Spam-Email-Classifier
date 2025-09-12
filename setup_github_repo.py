import subprocess
import os

# Set your GitHub repo URL here
GITHUB_URL = "https://github.com/iamdpsingh/Spam-Email-Classifier.git"  # <-- Replace this

# File types to be tracked by Git LFS
LFS_FILE_TYPES = ["*.csv", "*.pkl", "*.h5", "*.zip", "*.pt"]

# File types to ignore (not upload)
IGNORE_FILE_TYPES = ["*.db"]

def run(cmd):
    print(f"⚙️ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def update_gitignore():
    print("📝 Updating .gitignore with ignored file types...")
    with open(".gitignore", "a") as gitignore:
        for filetype in IGNORE_FILE_TYPES:
            gitignore.write(f"{filetype}\n")

def setup_github_repo():
    run("git init")
    run("git lfs install")

    for filetype in LFS_FILE_TYPES:
        run(f'git lfs track "{filetype}"')

    update_gitignore()

    run("git add .gitattributes")
    run("git add .gitignore")
    run("git add .")
    run('git commit -m "Initial commit with LFS and .gitignore" || echo \"Nothing to commit\"')

    run("git remote remove origin || true")
    run(f"git remote add origin {GITHUB_URL}")
    run("git branch -M main")
    run("git push -u origin main --force")

if __name__ == "__main__":
    setup_github_repo()
