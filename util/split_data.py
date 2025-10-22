import os, shutil, random

def make_subset(dir_name, data_dir, pct, categories):
    if dir_name.exists():
        print(f'ℹ️ {dir_name} already in dataset')
    else:
        for category in categories:
            src_dir = data_dir / category
            dst_dir = dir_name / category
            os.makedirs(dst_dir, exist_ok=True)
            all_files = list(src_dir.glob('*.jpg'))
            random.shuffle(all_files)
            n_images = int(len(all_files) * pct)
            selected_files = all_files[:n_images]
            
            for file_path in selected_files:
                shutil.move(file_path, dst_dir / file_path.name)
                
            print(f"✅ {category} {dir_name}: {len(selected_files)} data splitted")


def remove_dir(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        try:
            shutil.rmtree(dir_path)
            print(f"🗑️ {dir_path} deleted")
        except Exception as e:
            print(f"⚠️ could not delete {dir_path}: {e}")
    else:
        print(f"ℹ️ {dir_path} does not exists or it isn't a directory")