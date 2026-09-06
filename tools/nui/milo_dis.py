#!/usr/bin/env python3
"""Disassemble PowerPC code from the raw (pre-XEX) Milo PE image.

Usage:
  python milo_dis.py ADDR [COUNT]           # disassemble COUNT instructions at guest ADDR
  python milo_dis.py --bytes ADDR [LEN]     # hex dump
  python milo_dis.py --func NAME [COUNT]    # disassemble from symbol (map lookup)
"""
import sys
import os
import struct

import pefile
from capstone import Cs, CS_ARCH_PPC, CS_MODE_32, CS_MODE_BIG_ENDIAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milo_sym  # noqa: E402

DEFAULT_EXE = r"C:\Users\ellio\Documents\github\xenia-canary\xbe\Project Milo (May 17 2010)\miloReleaseLIB.exe"


class Image:
    def __init__(self, path=DEFAULT_EXE):
        self.pe = pefile.PE(path, fast_load=True)
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.data = self.pe.__data__
        self.sections = []
        for s in self.pe.sections:
            self.sections.append((self.base + s.VirtualAddress, s.Misc_VirtualSize,
                                  s.PointerToRawData, s.SizeOfRawData, s.Name.rstrip(b"\0").decode()))

    def read(self, va, n):
        out = bytearray()
        while n > 0:
            for (sva, vsize, raw, rawsize, name) in self.sections:
                if sva <= va < sva + max(vsize, rawsize):
                    off = va - sva
                    avail = rawsize - off
                    if avail <= 0:
                        out += b"\0" * min(n, vsize - off)
                        n -= min(n, vsize - off)
                        va += min(n, vsize - off)
                        break
                    take = min(n, avail)
                    out += self.data[raw + off: raw + off + take]
                    va += take
                    n -= take
                    break
            else:
                raise ValueError("va %08x not in image" % va)
        return bytes(out)


def main(argv):
    img = Image()
    syms = milo_sym.load_map(milo_sym.DEFAULT_MAP)
    md = Cs(CS_ARCH_PPC, CS_MODE_32 | CS_MODE_BIG_ENDIAN)
    md.detail = False
    if argv[0] == "--bytes":
        va = int(argv[1], 16)
        n = int(argv[2], 0) if len(argv) > 2 else 64
        b = img.read(va, n)
        for i in range(0, len(b), 16):
            print("%08x: %s" % (va + i, " ".join("%02x" % c for c in b[i:i + 16])))
        return
    if argv[0] == "--func":
        name = argv[1]
        va = None
        for rva, sname, obj in syms:
            if sname == name:
                va = rva
                break
        if va is None:
            print("symbol not found")
            return
        count = int(argv[2], 0) if len(argv) > 2 else 64
    else:
        va = int(argv[0], 16)
        count = int(argv[1], 0) if len(argv) > 1 else 32
    b = img.read(va, count * 4)
    for i in range(count):
        a = va + i * 4
        word = b[i * 4:i * 4 + 4]
        r = milo_sym.symbolize(syms, a)
        if r and r[1] == 0:
            print("\n%s:" % r[0])
        insns = list(md.disasm(word, a))
        if insns:
            ins = insns[0]
            extra = ""
            if ins.mnemonic.startswith("b") and ins.op_str:
                try:
                    tgt = int(ins.op_str.split(",")[-1].strip(), 16)
                    t = milo_sym.symbolize(syms, tgt)
                    if t:
                        extra = "   ; %s+0x%x" % (t[0], t[1])
                except ValueError:
                    pass
            print("%08x  %s  %-8s %s%s" % (a, word.hex(), ins.mnemonic, ins.op_str, extra))
        else:
            print("%08x  %s  .long    0x%s" % (a, word.hex(), word.hex()))


if __name__ == "__main__":
    main(sys.argv[1:])
