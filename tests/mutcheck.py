"""
Verify a test actually fails when the fix it guards is broken.

    python tests/mutcheck.py <src_file> <test_path> <old_text> <new_text>

Applies one mutation, runs the tests, prints CATCHES IT or !! VACUOUS !!, and
restores the file in a ``finally`` so an interrupt cannot leave the mutation on
disk. A vacuous result means the new test proves nothing and needs rewriting --
several tests in this suite only exist because this script said so.

Not a pytest file (no ``test_`` prefix): it is a development tool, kept in the
repo because temp directories keep getting cleared between sessions.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    src, tests, old, new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    path = Path(src)
    original = path.read_text(encoding="utf-8")
    count = original.count(old)
    if count != 1:
        print(f"SKIP   anchor appears {count}x")
        return 0
    # first thing's first: does the test even pass as things stand? a test
    # that's already red "catches" every mutation you throw at it, and that
    # fooled us once (the union widget test was failing on its own)
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", *tests.split()],
        capture_output=True, text=True, timeout=400,
    )
    if baseline.returncode != 0:
        tail = baseline.stdout.strip().split("\n")[-1] if baseline.stdout.strip() else "(no output)"
        print("!! BASELINE FAILS !! ", tail)
        return 2
    try:
        path.write_text(original.replace(old, new), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", *tests.split()],
            capture_output=True, text=True, timeout=400,
        )
        tail = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "(no output)"
        # careful: "no tests ran" also exits non-zero, and that once looked like a
        # catch to us. a real catch means failures, not pytest failing to collect
        caught = r.returncode != 0 and " failed" in tail
        print(("CATCHES IT   " if caught else "!! VACUOUS !! "), tail)
        return 0
    finally:
        path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
