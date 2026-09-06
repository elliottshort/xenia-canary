#!/usr/bin/env python3
"""Find code references to a guest address in the Milo PE (lis/addi|lwz|stw pairs, and branches).

Usage:
  python milo_xref.py ADDR            # data xrefs (lis hi + lo-half immediates)
  python milo_xref.py --calls ADDR    # bl/b targets == ADDR
  python milo_xref.py --str SUBSTR    # find strings containing SUBSTR (returns VA)
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milo_sym  # noqa: E402
import milo_dis  # noqa: E402


def text_ranges(img):
    for (sva, vsize, raw, rawsize, name) in img.sections:
        if name.startswith(".text"):
            yield sva, img.data[raw:raw + rawsize]


def find_data_xrefs(img, syms, target):
    hi = ((target + 0x8000) >> 16) & 0xffff
    lo = target & 0xffff
    lo_signed = lo - 0x10000 if lo & 0x8000 else lo
    results = []
    for sva, code in text_ranges(img):
        n = len(code) // 4
        words = struct.unpack(">%dI" % n, code[:n * 4])
        for i, w in enumerate(words):
            # lis rD, hi  == addis rD, 0, hi  (opcode 15, rA=0)
            if (w >> 26) == 15 and ((w >> 16) & 0x1f) == 0 and (w & 0xffff) == hi:
                rd = (w >> 21) & 0x1f
                # look ahead up to 12 instrs for an instruction using rd as rA with imm lo
                for j in range(i + 1, min(i + 13, n)):
                    w2 = words[j]
                    op = w2 >> 26
                    ra = (w2 >> 16) & 0x1f
                    if op in (14, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 62) and ra == rd and (w2 & 0xffff) == lo:
                        results.append(sva + j * 4)
                        break
                    # stop if rd is overwritten by another lis
                    if (w2 >> 26) == 15 and ((w2 >> 21) & 0x1f) == rd:
                        break
    return results


def find_calls(img, syms, target):
    results = []
    for sva, code in text_ranges(img):
        n = len(code) // 4
        words = struct.unpack(">%dI" % n, code[:n * 4])
        for i, w in enumerate(words):
            if (w >> 26) == 18:  # b/bl
                li = w & 0x03fffffc
                if li & 0x02000000:
                    li -= 0x04000000
                aa = w & 2
                tgt = li if aa else (sva + i * 4 + li)
                if tgt == target:
                    results.append(sva + i * 4)
    return results


def find_strings(img, needle):
    needle_b = needle.encode()
    out = []
    for (sva, vsize, raw, rawsize, name) in img.sections:
        data = img.data[raw:raw + rawsize]
        idx = 0
        while True:
            idx = data.find(needle_b, idx)
            if idx < 0:
                break
            # walk back to string start
            s = idx
            while s > 0 and 0x20 <= data[s - 1] < 0x7f:
                s -= 1
            e = idx
            while e < len(data) and 0x20 <= data[e] < 0x7f:
                e += 1
            out.append((sva + s, data[s:e].decode("ascii", "replace")))
            idx = e
    return out


def main(argv):
    img = milo_dis.Image()
    syms = milo_sym.load_map(milo_sym.DEFAULT_MAP)
    if argv[0] == "--str":
        for va, s in find_strings(img, argv[1]):
            print("%08x  %r" % (va, s))
        return
    if argv[0] == "--calls":
        target = int(argv[1], 16)
        refs = find_calls(img, syms, target)
    else:
        target = int(argv[0], 16)
        refs = find_data_xrefs(img, syms, target)
    for r in refs:
        s = milo_sym.symbolize(syms, r)
        print("%08x  in %s+0x%x" % (r, s[0], s[1]) if s else "%08x" % r)


if __name__ == "__main__":
    main(sys.argv[1:])
