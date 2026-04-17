from pathlib import Path
root = Path('ML-DL Mastery')
all_dirs = sorted([p for p in root.rglob('*') if p.is_dir()])
leaf_dirs = [p for p in all_dirs if not any(q.parent == p for q in all_dirs if q != p)]
print('total dirs', len(all_dirs))
print('leaf dirs', len(leaf_dirs))
for p in leaf_dirs[:40]:
    print(p.relative_to(root))
