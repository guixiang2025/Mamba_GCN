#!/usr/bin/env python3
"""
Project Structure Organizer for Mamba-GCN
Organizes project files into proper directory structure
"""

import os
import shutil
from pathlib import Path


def create_directory_structure():
    """Create the standard project directory structure"""

    directories = [
        'docs',           # Documentation files
        'scripts',        # Utility scripts
        'experiments',    # Experiment logs and results
        'tests',          # Test files
        'checkpoints',    # Model checkpoints
        'outputs',        # Output files
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")


def organize_files():
    """Organize files into appropriate directories"""

    print("\n📋 Organizing files...")

    # File organization rules
    file_moves = {
        # Documentation files
        'docs/': [
            'PRD.md',
            'PRD2.md',
            'DAY1_CHECKPOINT.md',
            'DAY1_MORNING_COMPLETION_SUMMARY.md',
            'ENVIRONMENT_SETUP_COMPLETE.md',
            'TASK_*.md',
            'ToDoLists.md'
        ],

        # Test files
        'tests/': [
            'test_*.py',
            'baseline_validation.py',
            'end_to_end_validation.py',
            'error_handling_validation.py',
            'poc_training_validation.py'
        ],

        # Scripts
        'scripts/': [
            'organize_project.py',
            'train_mock.py',
            'example_usage.py'
        ],

        # Experiment logs
        'experiments/': [
            '*.json',
            'poc_training_logs/'
        ]
    }

    # Move files according to rules
    for target_dir, patterns in file_moves.items():
        for pattern in patterns:
            if pattern.endswith('/'):
                # Directory
                source_path = Path(pattern.rstrip('/'))
                if source_path.exists() and source_path.is_dir():
                    target_path = Path(target_dir) / source_path.name
                    if not target_path.exists():
                        shutil.move(str(source_path), str(target_path))
                        print(f"   📂 {pattern} → {target_dir}")
            else:
                # File pattern
                if '*' in pattern:
                    # Glob pattern
                    for file_path in Path('.').glob(pattern):
                        if file_path.is_file():
                            target_path = Path(target_dir) / file_path.name
                            if not target_path.exists():
                                shutil.move(str(file_path), str(target_path))
                                print(f"   📄 {file_path.name} → {target_dir}")
                else:
                    # Exact file
                    source_path = Path(pattern)
                    if source_path.exists() and source_path.is_file():
                        target_path = Path(target_dir) / source_path.name
                        if not target_path.exists():
                            shutil.move(str(source_path), str(target_path))
                            print(f"   📄 {pattern} → {target_dir}")


def create_project_info():
    """Create project information files"""

    print("\n📋 Creating project info files...")

    # Create LICENSE file
    license_content = """MIT License

Copyright (c) 2025 Mamba-GCN Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    with open('LICENSE', 'w') as f:
        f.write(license_content)
    print("   ✅ LICENSE")

    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
*.ckpt

# Data files
*.npz
*.pkl
*.h5
*.hdf5

# Logs
*.log
logs/
wandb/
tensorboard_logs/

# Checkpoints
checkpoints/*.pth
checkpoints/*.pt

# Outputs
outputs/
results/
experiments/*.json
experiments/*.png
experiments/*.gif

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
~*
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("   ✅ .gitignore")


def print_final_structure():
    """Print the final project structure"""

    print("\n📁 Final Project Structure:")
    print("=" * 50)

    def print_tree(directory, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return

        path = Path(directory)
        if not path.exists():
            return

        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "

            if item.is_dir() and not item.name.startswith('.'):
                print(f"{prefix}{current_prefix}📂 {item.name}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
            elif item.is_file() and not item.name.startswith('.'):
                if item.suffix == '.py':
                    icon = "🐍"
                elif item.suffix == '.md':
                    icon = "📚"
                elif item.suffix == '.txt':
                    icon = "📄"
                elif item.suffix in ['.json', '.yaml', '.yml']:
                    icon = "⚙️"
                else:
                    icon = "📄"
                print(f"{prefix}{current_prefix}{icon} {item.name}")

    print_tree(".")


def main():
    """Main function to organize the project"""

    print("🗂️  MotionAGFormer + MambaGCN Project Organizer")
    print("=" * 50)

    # Create directory structure
    create_directory_structure()

    # Organize files
    organize_files()

    # Create project info files
    create_project_info()

    # Print final structure
    print_final_structure()

    print("\n✅ Project organization completed!")
    print("\n💡 Tips:")
    print("   - Run tests with: python -m pytest tests/")
    print("   - Check documentation in: docs/")
    print("   - View experiment logs in: experiments/")
    print("   - Install dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
