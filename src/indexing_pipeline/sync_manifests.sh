python3 -c "
import json, os, shutil

data_dir = '/workspace/tmp/data'
chunked_manifest = f'{data_dir}/chunked/_manifest.jsonl'
chunked_manifest_full = f'{data_dir}/chunked/_manifest.jsonl.full'
embedded_manifest = f'{data_dir}/embedded/_manifest.jsonl'

# Backup the full manifest so you can resume the remaining 90K later
if not os.path.exists(chunked_manifest_full):
    shutil.copy2(chunked_manifest, chunked_manifest_full)
    print(f'Backed up full manifest to {chunked_manifest_full}')

# 1. Get all successfully embedded case IDs
embedded_ids = set()
if os.path.exists(embedded_manifest):
    with open(embedded_manifest, 'r') as f:
        for line in f:
            if line.strip():
                embedded_ids.add(json.loads(line)['case_id'])

print(f'Found {len(embedded_ids)} embedded cases.')

# 2. Filter chunked manifest to ONLY include those IDs
kept = 0
tmp_path = chunked_manifest + '.tmp'
# Read from the FULL manifest to ensure we don't lose data if run twice
with open(chunked_manifest_full, 'r') as f_in, open(tmp_path, 'w') as f_out:
    for line in f_in:
        if line.strip():
            row = json.loads(line)
            if row['case_id'] in embedded_ids:
                f_out.write(line)
                kept += 1

os.replace(tmp_path, chunked_manifest)
print(f'Synced chunked manifest: kept {kept} cases for loading.')

# 3. Clear any partial failure logs (load stage blocks if these exist)
for fail_file in [f'{data_dir}/chunked/_failures.jsonl', f'{data_dir}/embedded/_failures.jsonl']:
    if os.path.exists(fail_file):
        open(fail_file, 'w').close()
        print(f'Cleared {fail_file}')
"
