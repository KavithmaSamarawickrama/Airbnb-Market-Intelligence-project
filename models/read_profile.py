import json

with open("profile_report.json", "rb") as f:
    raw_content = f.read()

content = raw_content.decode("utf-16")
lines = content.splitlines()

non_log_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped:
        continue
    try:
        obj = json.loads(stripped)
        if "level" in obj and "logger" in obj and "message" in obj:
            # It's a log line, skip it
            continue
    except Exception:
        pass
    non_log_lines.append(line)

json_str = "\n".join(non_log_lines)
try:
    data = json.loads(json_str)
    print("Successfully parsed profiling JSON!")
    print("\nProfiles found:")
    for prof in data.get("profiles", []):
        print(f"\n- Table: {prof.get('table')}")
        print(f"  Rows: {prof.get('total_rows')}")
        cols = prof.get('columns', [])
        print(f"  Columns: {len(cols)}")
        
        # Look at null rates
        high_nulls = [c for c in cols if c.get('null_rate', 0) > 0]
        if high_nulls:
            print("  Columns with Nulls:")
            for c in sorted(high_nulls, key=lambda x: x['null_rate'], reverse=True)[:5]:
                print(f"    * {c['name']} ({c['type']}): {c['null_count']} nulls ({c['null_rate']:.2%})")
        
        # Check validation issues
        issues = prof.get('validation_issues', [])
        print(f"  Validation Issues: {len(issues)}")
        for issue in issues[:3]:
            print(f"    * Warning: {issue}")
except Exception as e:
    print("Failed to parse JSON:", e)
    print("\nFirst 15 non-log lines:")
    for l in non_log_lines[:15]:
        print(l)
