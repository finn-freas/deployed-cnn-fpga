#!/bin/bash
set -e
export PATH="$HOME/oss-cad-suite/bin:$PATH"
D=$1; P=$2; OUTFS=$3
python3 - "$D/pnr.json" "$P" "$D/pnr_psda.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
n = 0
for mod in j["modules"].values():
    for c in mod["cells"].values():
        if c["type"] == "rPLL":
            old = c["parameters"]["PSDA_SEL"]
            c["parameters"]["PSDA_SEL"] = format(int(sys.argv[2]), "04b") + (" " if old.endswith(" ") else "")
            n += 1
assert n == 1, n
json.dump(j, open(sys.argv[3], "w"))
EOF
gowin_pack -d GW2A-18C -o "$OUTFS" "$D/pnr_psda.json" 2>&1 | grep -v "UserWarning\|warnings.warn" || true
rm -f "$D/pnr_psda.json"
ls -la "$OUTFS"
