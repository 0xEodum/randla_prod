from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import warnings

ROOT_PATH = (Path(__file__) / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'datasets' / 's3dis'
RAW_PATH = DATASET_PATH / 'Stanford3dDataset_v1.2_Aligned_Version'
LABELS_PATH = DATASET_PATH / 'classes.json'
TRAIN_PATH = DATASET_PATH / 'train'
TEST_PATH = DATASET_PATH / 'test'
VAL_PATH = DATASET_PATH / 'val'

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
    folder.mkdir(exist_ok=True)

if LABELS_PATH.exists():
    print(LABELS_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels_dict = defaultdict(lambda: len(labels_dict.keys()), json.load(f))
else:
    labels_dict = defaultdict(lambda: len(labels_dict.keys()))

def load_annotation_file(file_path, max_retries=3):
    """Load annotation file with robust error handling."""
    for encoding in ['utf-8', 'latin1', 'ascii']:
        for attempt in range(max_retries):
            try:
                print(f"Trying {file_path.name} with encoding {encoding}, attempt {attempt + 1}")
                
                # Try different loading methods
                if attempt == 0:
                    # Standard numpy loadtxt
                    return np.loadtxt(file_path, dtype=np.float32)
                elif attempt == 1:
                    # Manual parsing with error handling
                    return load_with_error_skip(file_path, encoding)
                else:
                    # Pandas fallback with error handling
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path, sep=r'\s+', header=None, encoding=encoding, 
                                       on_bad_lines='skip', engine='python')
                        return df.values.astype(np.float32)
                    except ImportError:
                        continue
                        
            except (ValueError, UnicodeDecodeError, IOError) as e:
                print(f"  Failed with {encoding}, attempt {attempt + 1}: {e}")
                continue
    
    raise ValueError(f"Could not load file {file_path} with any method")

def load_with_error_skip(file_path, encoding='utf-8'):
    """Load file line by line, skipping corrupted lines."""
    valid_lines = []
    corrupted_count = 0
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Try to parse the line
                    values = line.strip().split()
                    if len(values) >= 6:  # x y z r g b [label]
                        float_values = [float(v) for v in values]
                        valid_lines.append(float_values)
                except (ValueError, IndexError):
                    corrupted_count += 1
                    if corrupted_count <= 10:  # Print first 10 errors
                        print(f"  Skipping corrupted line {line_num}: {line.strip()[:100]}")
    except UnicodeDecodeError as e:
        print(f"  Encoding error with {encoding}: {e}")
        raise
    
    if not valid_lines:
        raise ValueError(f"No valid lines found in {file_path}")
    
    if corrupted_count > 0:
        print(f"  Skipped {corrupted_count} corrupted lines")
    
    return np.array(valid_lines, dtype=np.float32)

for area_number in range(1, 7):
    print(f'Processing point clouds of area {area_number:d}')
    dir = RAW_PATH / f'Area_{area_number:d}'
    if not dir.exists():
        warnings.warn(f'Area {area_number:d} not found')
        continue
        
    for pc_path in sorted(list(dir.iterdir())):
        if not pc_path.is_dir():
            continue
            
        pc_name = f'{area_number:d}_' + pc_path.stem + '.npy'

        # Check if point cloud has already been processed
        if list(ROOT_PATH.rglob(pc_name)):
            continue

        points_list = []
        annotation_path = pc_path / 'Annotations'
        if not annotation_path.exists():
            print(f'No Annotations folder found for {pc_path.name}')
            continue
            
        for elem in sorted(list(annotation_path.glob('*.txt'))):
            label = elem.stem.split('_')[0]
            print(f'Processing {pc_name}: adding {label} to point cloud...          ', end='\r')
            
            try:
                points = load_annotation_file(elem)
                if points.size == 0:
                    print(f"\nWarning: Empty file {elem}")
                    continue
                    
                label_id = labels_dict[label]
                
                # Ensure correct shape (N, 6) -> (N, 7)
                if points.shape[1] == 6:  # x y z r g b
                    labelled_points = np.column_stack((points, np.full(points.shape[0], label_id)))
                else:
                    print(f"\nWarning: Unexpected shape {points.shape} in {elem}")
                    continue
                    
                points_list.append(labelled_points.astype(np.float32))
                
            except Exception as e:
                print(f"\nError processing {elem}: {e}")
                print("Skipping this file...")
                continue

        if not points_list:
            print(f"\nNo valid annotation files found for {pc_name}")
            continue

        # Save updated labels dict
        with open(LABELS_PATH, 'w') as f:
            json.dump(dict(labels_dict), f, indent=2)

        # Merge all subclouds together
        try:
            merged_points = np.vstack(points_list)
            print(f"\nMerged {len(points_list)} annotation files into {merged_points.shape[0]} points")
        except Exception as e:
            print(f"\nError merging points for {pc_name}: {e}")
            continue

        # Save computed point cloud
        path = TRAIN_PATH if area_number < 5 else TEST_PATH
        try:
            np.save(path / pc_name, merged_points, allow_pickle=False)
            print(f"Saved {pc_name} to {path}")
        except Exception as e:
            print(f"Error saving {pc_name}: {e}")

print('Done.')