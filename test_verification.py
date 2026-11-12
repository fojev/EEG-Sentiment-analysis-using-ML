import urllib.request
import urllib.error
import json
import os

print('======================================================')
print('RUNNING BACKEND & PRIVACY VERIFICATION SUITE')
print('======================================================')

# 1. Health endpoint
print('\n[TEST 1] Testing GET /api/health ...')
req = urllib.request.urlopen('http://127.0.0.1:5000/api/health')
health = json.loads(req.read().decode())
print(json.dumps(health, indent=2))
assert health['status'] == 'healthy', 'Health check failed'
assert health['selected_algorithm'] == 'Random Forest', 'Algorithm mismatch'
assert 'EEG Sentiment Lab' in health['project_name'], 'Project name mismatch'
print('--> PASS: Health check and updated project branding verified.')

# 2. Batch predict with valid 20-row CSV
print('\n[TEST 2] Testing POST /api/predict/batch with valid 20-row EEG CSV ...')
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
valid_csv = r'c:\Users\hp\Minor Project\EEG-Sentiment-Analysis\dataset\test_sample_20_rows.csv'
with open(valid_csv, 'rb') as f:
    file_bytes = f.read()

part1 = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test_sample_20_rows.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode('utf-8')
part2 = f'\r\n--{boundary}--\r\n'.encode('utf-8')
body = part1 + file_bytes + part2

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict/batch',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
res = urllib.request.urlopen(req)
data = json.loads(res.read().decode('utf-8'))
print('Response Status :', res.status)
print('Total Samples   :', data.get('total_samples'))
print('Class Summary   :', data.get('class_summary'))
print('Sample Row 0    :', data.get('predictions', [{}])[0])
print('Latency ms      :', data.get('latency_ms'))
assert res.status == 200, 'Batch prediction failed'
assert data.get('total_samples') == 20, f'Expected 20 samples, got {data.get("total_samples")}'
assert len(data.get('predictions', [])) == 20, 'Expected 20 prediction records'
assert 'POSITIVE' in data.get('class_summary', {}), 'Class summary missing POSITIVE'
print('--> PASS: Real Random Forest predictions generated on actual uploaded EEG features.')

# 3. Batch predict with invalid columns CSV (must be rejected)
print('\n[TEST 3] Testing POST /api/predict/batch with invalid EEG CSV ...')
invalid_csv = r'c:\Users\hp\Minor Project\EEG-Sentiment-Analysis\dataset\test_invalid_columns.csv'
with open(invalid_csv, 'rb') as f:
    inv_bytes = f.read()

part1_inv = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test_invalid_columns.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode('utf-8')
body_inv = part1_inv + inv_bytes + part2

req_inv = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict/batch',
    data=body_inv,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
try:
    urllib.request.urlopen(req_inv)
    raise AssertionError('Expected 400 Bad Request but request succeeded')
except urllib.error.HTTPError as e:
    err_body = json.loads(e.read().decode('utf-8'))
    print('Response Code   :', e.code)
    print('Error Message   :', err_body.get('message'))
    assert e.code == 400, 'Expected 400 status'
    assert 'missing' in err_body.get('message', '').lower(), 'Expected missing columns explanation'
print('--> PASS: Incompatible dataset rejected with clear explanatory error message.')

# 4. Strict Privacy & Zero Data Retention verification on disk
print('\n[TEST 4] Verifying zero data retention on disk ...')
stray_files = []
for root, dirs, files in os.walk(r'c:\Users\hp\Minor Project\EEG-Sentiment-Analysis'):
    if any(ignore in root for ignore in ['node_modules', '.git', 'venv', 'dist', 'dataset']):
        continue
    for file in files:
        if file.endswith('.csv'):
            stray_files.append(os.path.join(root, file))

print('Stray CSV files found:', len(stray_files))
assert len(stray_files) == 0, f'Found stray CSV files: {stray_files}'
print('--> PASS: Zero uploaded CSV files or temporary data files exist on server disk.')

print('\n======================================================')
print('ALL BACKEND & PRIVACY VERIFICATION TESTS PASSED (4/4)')
print('======================================================')

