import subprocess
import os
import time
import datetime
import schedule
from pathlib import Path

# Set your GitHub repo URL here
GITHUB_URL = "https://github.com/iamdpsingh/Spam-Email-Classifier.git"  # <-- Replace this

# File types to be tracked by Git LFS
LFS_FILE_TYPES = ["*.csv", "*.pkl", "*.h5", "*.zip", "*.pt", "*.joblib", "*.model"]

# File types to ignore (not upload)
IGNORE_FILE_TYPES = ["*.db", "*.log", "*.tmp", "__pycache__/", "*.pyc"]

# Auto-commit settings
AUTO_COMMIT_TIME = "23:59"  # Time to trigger daily commit (11:59 PM)
COMMIT_MESSAGE_PREFIX = "🚀 Daily Auto-Commit"

class GitHubAutoCommitter:
    def __init__(self):
        self.last_commit_date = None
        self.setup_complete = False
        
    def run(self, cmd):
        """Execute shell command with logging"""
        print(f"⚙️ Executing: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            if result.stdout:
                print(f"✅ Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"❌ Error details: {e.stderr.strip()}")
            return False

    def update_gitignore(self):
        """Update .gitignore with ignored file types"""
        print("📝 Updating .gitignore with ignored file types...")
        
        gitignore_content = []
        
        # Read existing .gitignore if it exists
        if os.path.exists('.gitignore'):
            with open('.gitignore', 'r') as f:
                gitignore_content = f.read().splitlines()
        
        # Add new ignore patterns
        for filetype in IGNORE_FILE_TYPES:
            if filetype not in gitignore_content:
                gitignore_content.append(filetype)
        
        # Add common Python ignores
        python_ignores = [
            "# Python",
            "*.pyc",
            "__pycache__/",
            "*.pyo",
            "*.pyd",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "# Virtual environments",
            "venv/",
            "env/",
            "ENV/",
            "spam_classifier_env/",
            "# IDE",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "# OS",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        for ignore in python_ignores:
            if ignore not in gitignore_content:
                gitignore_content.append(ignore)
        
        # Write updated .gitignore
        with open('.gitignore', 'w') as f:
            f.write('\n'.join(gitignore_content) + '\n')

    def setup_github_repo(self):
        """Initial repository setup"""
        print("🚀 Setting up GitHub repository...")
        
        # Initialize git if not already done
        if not os.path.exists('.git'):
            self.run("git init")
        
        # Setup Git LFS
        self.run("git lfs install")
        
        # Track large file types with LFS
        for filetype in LFS_FILE_TYPES:
            self.run(f'git lfs track "{filetype}"')
        
        # Update .gitignore
        self.update_gitignore()
        
        # Add and commit initial files
        self.run("git add .gitattributes")
        self.run("git add .gitignore")
        self.run("git add .")
        
        # Initial commit
        self.run('git commit -m "🎯 Initial commit: Spam Email Classifier Setup" || echo "Nothing to commit"')
        
        # Setup remote
        self.run("git remote remove origin 2>/dev/null || true")
        self.run(f"git remote add origin {GITHUB_URL}")
        self.run("git branch -M main")
        
        # Push to remote
        if self.run("git push -u origin main --force"):
            print("✅ Repository setup completed successfully!")
            self.setup_complete = True
        else:
            print("❌ Failed to push to remote repository")

    def check_for_changes(self):
        """Check if there are any changes to commit"""
        try:
            result = subprocess.run("git status --porcelain", shell=True, 
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip()) > 0
        except:
            return False

    def get_file_stats(self):
        """Get statistics about changed files"""
        try:
            # Get added/modified files
            result = subprocess.run("git diff --name-status HEAD", shell=True, 
                                  capture_output=True, text=True)
            changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get untracked files
            result = subprocess.run("git ls-files --others --exclude-standard", shell=True,
                                  capture_output=True, text=True)
            untracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return len(changes), len(untracked)
        except:
            return 0, 0

    def daily_commit(self):
        """Perform daily automatic commit"""
        current_date = datetime.date.today()
        
        print(f"\n🕒 Daily commit trigger at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.setup_complete:
            print("⚠️ Repository not setup yet, performing initial setup...")
            self.setup_github_repo()
            return
        
        # Check for changes
        if not self.check_for_changes():
            print("📄 No changes detected, skipping commit")
            return
        
        # Get file statistics
        modified_count, untracked_count = self.get_file_stats()
        
        # Create commit message with file stats
        commit_message = f"{COMMIT_MESSAGE_PREFIX} - {current_date.strftime('%Y-%m-%d')}"
        
        if modified_count > 0 or untracked_count > 0:
            commit_message += f" | Modified: {modified_count}, New: {untracked_count} files"
        
        # Add all changes
        print("📂 Adding all changes...")
        self.run("git add .")
        
        # Commit changes
        print(f"💾 Committing with message: {commit_message}")
        if self.run(f'git commit -m "{commit_message}"'):
            # Push to remote
            print("🚀 Pushing to GitHub...")
            if self.run("git push origin main"):
                print("✅ Daily commit completed successfully!")
                self.last_commit_date = current_date
                
                # Log the commit
                self.log_commit(commit_message)
            else:
                print("❌ Failed to push to GitHub")
        else:
            print("❌ Failed to commit changes")

    def log_commit(self, message):
        """Log commit information"""
        log_file = "auto_commit_log.txt"
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        
        print(f"📝 Logged commit to {log_file}")

    def manual_commit(self, message=None):
        """Manual commit function"""
        if not message:
            message = f"Manual commit - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print(f"🔧 Manual commit: {message}")
        
        if self.check_for_changes():
            self.run("git add .")
            if self.run(f'git commit -m "{message}"'):
                if self.run("git push origin main"):
                    print("✅ Manual commit completed!")
                    self.log_commit(message)
        else:
            print("📄 No changes to commit")

    def start_scheduler(self):
        """Start the automatic commit scheduler"""
        print(f"🕒 Starting scheduler - Daily commits at {AUTO_COMMIT_TIME}")
        print("📋 Scheduler will check for changes and commit automatically")
        print("🛑 Press Ctrl+C to stop the scheduler")
        
        # Schedule daily commit
        schedule.every().day.at(AUTO_COMMIT_TIME).do(self.daily_commit)
        
        # Optional: Also schedule every hour to check for immediate commits
        # Uncomment the next line if you want hourly checks
        # schedule.every().hour.do(self.check_and_commit_if_needed)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped by user")

    def check_and_commit_if_needed(self):
        """Check and commit if significant changes detected"""
        if self.check_for_changes():
            modified_count, untracked_count = self.get_file_stats()
            
            # Only commit if significant changes (more than 3 files or important files)
            if modified_count + untracked_count >= 3:
                print(f"📊 Significant changes detected: {modified_count + untracked_count} files")
                self.daily_commit()

def main():
    """Main function with menu options"""
    committer = GitHubAutoCommitter()
    
    print("🏆 Spam Email Classifier - GitHub Auto Committer")
    print("=" * 50)
    
    while True:
        print("\n📋 Available Options:")
        print("1. 🔧 Setup GitHub Repository")
        print("2. 🚀 Start Daily Auto-Commit Scheduler")
        print("3. 💾 Manual Commit Now")
        print("4. 📊 Check Status")
        print("5. 🛑 Exit")
        
        choice = input("\n👉 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            committer.setup_github_repo()
            
        elif choice == '2':
            if not committer.setup_complete:
                print("⚠️ Setting up repository first...")
                committer.setup_github_repo()
            
            if committer.setup_complete:
                committer.start_scheduler()
            
        elif choice == '3':
            message = input("💬 Enter commit message (optional): ").strip()
            committer.manual_commit(message if message else None)
            
        elif choice == '4':
            if committer.check_for_changes():
                modified, untracked = committer.get_file_stats()
                print(f"📊 Status: Changes detected - Modified: {modified}, New: {untracked}")
            else:
                print("📄 Status: No changes detected")
                
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    # Install required package if not installed
    try:
        import schedule
    except ImportError:
        print("📦 Installing required package: schedule")
        subprocess.run("pip install schedule", shell=True, check=True)
        import schedule
    
    main()
