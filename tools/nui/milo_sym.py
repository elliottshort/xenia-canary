#!/usr/bin/env python3
"""Symbolize guest addresses using a Milo linker .map file.

Usage:
  python milo_sym.py [--map PATH] ADDR [ADDR ...]
  python milo_sym.py --find NAME_SUBSTRING
"""
import bisect
import os
import re
import sys

DEFAULT_MAP = r"C:\Users\ellio\Documents\github\xenia-canary\xbe\Project Milo (May 17 2010)\miloReleaseLIB.map"
# Produced by milo_pdb.py: VA \t size \t kind \t name \t module.  Adds the
# STATIC (module-local) functions the .map file does not list.
DEFAULT_PDB_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "milo_symbols.tsv")

_line_re = re.compile(r"^ ([0-9a-f]{4}):([0-9a-f]{8})\s+(\S+)\s+([0-9a-f]{8})\s+(f?)\s*(i?)\s*(\S*)\s*$")


def load_pdb_tsv(path=DEFAULT_PDB_TSV):
    """Load milo_symbols.tsv -> list of (rva, name, obj)."""
    syms = []
    if not path or not os.path.exists(path):
        return syms
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                rva = int(parts[0], 16)
            except ValueError:
                continue
            if rva == 0:
                continue
            module = parts[4] if len(parts) > 4 else ""
            syms.append((rva, parts[3], module))
    return syms


def load_map(path, pdb_tsv=DEFAULT_PDB_TSV):
    by_addr = {}  # rva -> (name, obj)
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _line_re.match(line.rstrip("\n"))
            if not m:
                continue
            rva = int(m.group(4), 16)
            if rva == 0:
                continue
            by_addr.setdefault(rva, (m.group(3), m.group(7)))
    # PDB symbols win at addresses present in both; map-only entries are kept.
    for rva, name, obj in load_pdb_tsv(pdb_tsv):
        by_addr[rva] = (name, obj)
    syms = [(rva, nm, ob) for rva, (nm, ob) in by_addr.items()]
    syms.sort()
    return syms


def symbolize(syms, addr):
    addrs = [s[0] for s in syms]
    i = bisect.bisect_right(addrs, addr) - 1
    if i < 0:
        return None
    rva, name, obj = syms[i]
    return name, addr - rva, obj


def main(argv):
    path = DEFAULT_MAP
    args = list(argv)
    if "--map" in args:
        i = args.index("--map")
        path = args[i + 1]
        del args[i:i + 2]
    syms = load_map(path)
    if args and args[0] == "--find":
        needle = args[1].lower()
        for rva, name, obj in syms:
            if needle in name.lower():
                print("%08x %s  [%s]" % (rva, name, obj))
        return
    for a in args:
        addr = int(a, 16)
        r = symbolize(syms, addr)
        if r is None:
            print("%08x: <no symbol>" % addr)
        else:
            print("%08x: %s+0x%x  [%s]" % (addr, r[0], r[1], r[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
