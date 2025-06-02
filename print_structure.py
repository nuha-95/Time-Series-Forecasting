import os

def print_repo_structure(start_path='.', prefix=''):
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if not e.startswith('.')]  # Ignore hidden files/folders
    for idx, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = '└── ' if idx == len(entries) - 1 else '├── '
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = '    ' if idx == len(entries) - 1 else '│   '
            print_repo_structure(path, prefix + extension)

if __name__ == '__main__':
    print_repo_structure()
