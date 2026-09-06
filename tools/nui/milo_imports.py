#!/usr/bin/env python3
"""Map XEX import records (thunk addresses) to ordinals and names.

Reads the unencrypted XEX2 header import table for record addresses, reads the
ordinal words from the raw PE image at those addresses, and names them via the
.map file (__imp_* symbols).

Usage: python milo_imports.py [lib_substring]
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milo_sym  # noqa: E402
import milo_dis  # noqa: E402

XEX = r"C:\Users\ellio\Documents\github\xenia-canary\xbe\Project Milo (May 17 2010)\miloReleaseLIB.xex"


def main(argv):
    want = argv[0].lower() if argv else ""
    d = open(XEX, "rb").read()
    module_flags, pe_off, _, sec_off, hdr_count = struct.unpack(">IIIII", d[4:24])
    keys = {}
    off = 24
    for _ in range(hdr_count):
        k, v = struct.unpack(">II", d[off:off + 8])
        off += 8
        keys[k] = v
    o = keys[0x000103FF]
    size, string_table_size, lib_count = struct.unpack(">III", d[o:o + 12])
    strtab = d[o + 12:o + 12 + string_table_size]
    # string table entries are NUL-terminated, padded to 4 bytes
    names = []
    p = 0
    while p < len(strtab):
        e = strtab.find(b"\0", p)
        if e < 0:
            break
        names.append(strtab[p:e].decode("ascii", "replace"))
        p = e + 1
        while p < len(strtab) and p % 4:
            p += 1
    q = o + 12 + string_table_size
    img = milo_dis.Image()
    syms = milo_sym.load_map(milo_sym.DEFAULT_MAP)
    impnames = {rva: name for rva, name, obj in syms if name.startswith("__imp_")}
    for _ in range(lib_count):
        lsize, = struct.unpack(">I", d[q:q + 4])
        lid, ver, minver, name_index, count = struct.unpack(">IIIHH", d[q + 24:q + 40])
        libname = names[name_index] if name_index < len(names) else "?"
        recs = struct.unpack(">%dI" % count, d[q + 40:q + 40 + 4 * count])
        if not want or want in libname.lower():
            print("== %s (%d imports)" % (libname, count))
            for r in recs:
                try:
                    val = struct.unpack(">I", img.read(r, 4))[0]
                except Exception:
                    val = 0
                ordinal = val & 0xFFFF
                typ = val >> 24
                nm = impnames.get(r, "")
                if nm == "":
                    s = milo_sym.symbolize(syms, r)
                    nm = "(%s+0x%x)" % (s[0], s[1]) if s else "?"
                print("  rec %08x val %08x ordinal 0x%03X type %d  %s" % (r, val, ordinal, typ, nm))
        q += lsize


if __name__ == "__main__":
    main(sys.argv[1:])
