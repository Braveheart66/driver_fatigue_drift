import json
p='data/processed/yawdd_audit.json'
with open(p,'r',encoding='utf-8') as f:
    j=json.load(f)
print('sessions:', len(j.get('sessions',{})))
print('summary:', j.get('summary'))
# also report mediapipe presence hint
print('Note: mean_landmark_count==null indicates mediapipe was unavailable during audit (landmark metrics skipped)')
