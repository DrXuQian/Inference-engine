#!/bin/bash
# Read model output_dir from split_meta.json
# Usage: MODEL=$(bash get_model_path.sh /path/to/model_dir)
META="$1/split_meta.json"
if [ -f "$META" ]; then
    python3 -c "import json; print(json.load(open('$META'))['output_dir'])" 2>/dev/null
fi
