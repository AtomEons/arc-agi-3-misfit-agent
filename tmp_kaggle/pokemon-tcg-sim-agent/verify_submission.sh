#!/usr/bin/env bash
# Triple-check verification for a PTCG submission tarball.
# Exits 0 if all four gates pass, nonzero on first failure.
set -u
TAR="${1:-submission.tar.gz}"
if [ ! -f "$TAR" ]; then echo "FAIL: tar not found: $TAR"; exit 1; fi

VERIFY_DIR=$(mktemp -d)
trap "rm -rf $VERIFY_DIR" EXIT

echo "GATE 1: tar extracts cleanly"
if ! tar -xzf "$TAR" -C "$VERIFY_DIR" 2>&1; then echo "FAIL gate 1"; exit 1; fi
echo "  PASS"

echo "GATE 2: tar contains main.py + at least one deck CSV + cg/"
INVENTORY=$(cd "$VERIFY_DIR" && find . -maxdepth 3 -type f | sort)
echo "$INVENTORY" | head -20
if ! echo "$INVENTORY" | grep -q "^\./main\.py$"; then echo "FAIL gate 2: no main.py"; exit 1; fi
if ! echo "$INVENTORY" | grep -qE "\./deck.*\.csv$"; then echo "FAIL gate 2: no deck CSV"; exit 1; fi
if ! echo "$INVENTORY" | grep -q "^\./cg/api\.py$"; then echo "FAIL gate 2: no cg/api.py"; exit 1; fi
echo "  PASS"

echo "GATE 3: main.py imports cleanly from inside extracted dir"
cd "$VERIFY_DIR"
IMPORT_OUT=$(python -c "import sys; sys.path.insert(0,'.'); import main; print('OK', hasattr(main,'agent'), hasattr(main,'read_deck_csv'))" 2>&1)
if [ $? -ne 0 ]; then echo "FAIL gate 3: import error"; echo "$IMPORT_OUT"; exit 1; fi
echo "  $IMPORT_OUT"
if ! echo "$IMPORT_OUT" | grep -q "OK True True"; then echo "FAIL gate 3: missing agent or read_deck_csv"; exit 1; fi
echo "  PASS"

echo "GATE 4: read_deck_csv() finds its deck file inside extracted dir"
DECK_OUT=$(python -c "import sys; sys.path.insert(0,'.'); import main; d = main.read_deck_csv(); print('deck_size', len(d), 'first', d[0] if d else None)" 2>&1)
if [ $? -ne 0 ]; then echo "FAIL gate 4: read_deck_csv error"; echo "$DECK_OUT"; exit 1; fi
echo "  $DECK_OUT"
if ! echo "$DECK_OUT" | grep -q "deck_size 60"; then echo "FAIL gate 4: deck not 60 cards"; exit 1; fi
echo "  PASS"

echo ""
echo "ALL FOUR GATES PASSED — tar is shipping-ready"
exit 0
