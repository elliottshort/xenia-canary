#!/usr/bin/env python3
"""Generate masked byte signatures for NUI runtime functions.

Reads PowerPC code from a raw image (the pre-XEX PE of Project Milo, or a
module image dumped by xenia) and emits, for each requested function, the
first N instruction words with masks that zero the fields that move between
builds of the same library (branch displacements and the immediates of
lis/addi/lwz/stw pairs that address globals).

Usage:
  python gen_signatures.py --exe IMAGE.exe --map IMAGE.map NAME [NAME ...]
      [--words N] [--format cc|toml]
  python gen_signatures.py --exe IMAGE.exe --addr 0x82C66C10 --name X ...

  python gen_signatures.py --verify IMAGE2.exe --map2 IMAGE2.map ...
      also checks that each signature matches exactly once in a second
      build and that the match is at the symbol of the same name.

Output "cc" is a C++ initializer for xe::kernel::nui::NuiFunctionSignature.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milo_dis  # noqa: E402
import milo_sym  # noqa: E402

BRANCH_MASK = 0xFC000003     # b/bl: keep opcode, AA, LK
CONDBR_MASK = 0xFFFF0003     # bc absolute: keep everything but BD
HIGH16_MASK = 0xFFFF0000     # keep opcode/registers, mask 16-bit immediate
EXACT = 0xFFFFFFFF

D_FORM_OPCODES = {14, 15, 24, 25, 26, 27, 28, 29,  # addi, addis, ori.., andi..
                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                  47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 62}


def mask_words(words):
    """Return (words, masks) with relocation-sensitive fields masked."""
    masks = []
    remembered = {}  # register -> instructions remaining
    for w in words:
        op = w >> 26
        mask = EXACT
        rd = (w >> 21) & 0x1F
        ra = (w >> 16) & 0x1F
        if op == 18:
            mask = BRANCH_MASK
        elif op == 16 and (w & 2):
            mask = CONDBR_MASK
        elif op == 15 and ra == 0:
            # lis rD, hi: the pair partner will use rD as rA (or rS for ori).
            mask = HIGH16_MASK
            remembered[rd] = 8
        elif op in D_FORM_OPCODES and ra in remembered:
            mask = HIGH16_MASK
        elif op == 24 and rd in remembered:  # ori rA, rS, lo (rS in rd slot)
            mask = HIGH16_MASK
        # Age remembered registers; a redefinition of rD forgets it.
        for reg in list(remembered):
            remembered[reg] -= 1
            if remembered[reg] <= 0:
                del remembered[reg]
        if op != 15 and rd in remembered and op in D_FORM_OPCODES and \
                op not in (24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 44, 45, 46,
                           47, 52, 53, 54, 55, 62):
            # Loads that overwrite rD end its use as a base.
            del remembered[rd]
        masks.append(mask)
    return words, masks


def read_words(img, va, count):
    data = img.read(va, count * 4)
    return list(struct.unpack(">%dI" % count, data))


def find_symbol(syms, name):
    for rva, sname, obj in syms:
        if sname == name:
            return rva
    return None


def count_matches(img, words, masks):
    """Count (mask-aware) matches over the .text sections."""
    n = len(words)
    hits = []
    for (sva, vsize, raw, rawsize, secname) in img.sections:
        if not secname.startswith(".text"):
            continue
        data = img.data[raw:raw + rawsize]
        total = len(data) // 4
        code = struct.unpack(">%dI" % total, data[:total * 4])
        first_w = words[0] & masks[0]
        first_m = masks[0]
        for i in range(total - n + 1):
            if (code[i] & first_m) != first_w:
                continue
            ok = True
            for j in range(1, n):
                if (code[i + j] & masks[j]) != (words[j] & masks[j]):
                    ok = False
                    break
            if ok:
                hits.append(sva + i * 4)
    return hits


def emit_cc(name, words, masks, address):
    def fmt(vals):
        return ", ".join("0x%08X" % v for v in vals)
    return ('      {"%s", "hle", 0x%08X,\n       {%s},\n       {%s}},'
            % (name, address, fmt(words), fmt(masks)))


def emit_toml(name, words, masks):
    pat = []
    for w, m in zip(words, masks):
        s = ""
        for nib in range(7, -1, -1):
            mn = (m >> (nib * 4)) & 0xF
            wn = (w >> (nib * 4)) & 0xF
            s += "%X" % wn if mn == 0xF else "?"
        pat.append(s)
    return '[[library.function]]\nname = "%s"\nmode = "hle"\npattern = "%s"\n' % (
        name, " ".join(pat))


class RawImage:
    """A module image dumped by xenia (--nui_dump_module_image) plus its
    .json sidecar; exposes the same read()/sections/data interface as
    milo_dis.Image so signatures can be generated from retail titles."""

    def __init__(self, bin_path, sidecar_path):
        import json
        self.data = open(bin_path, "rb").read()
        meta = json.load(open(sidecar_path, "r", encoding="utf-8"))
        self.base = int(meta["base"], 16)
        self.sections = []
        for r in meta.get("code_ranges", []):
            start = int(r["start"], 16)
            size = int(r["size"], 16)
            off = start - self.base
            self.sections.append((start, size, off, size, ".text"))

    def read(self, va, n):
        off = va - self.base
        if off < 0 or off + n > len(self.data):
            raise ValueError("va %08x not in image" % va)
        return self.data[off:off + n]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=milo_dis.DEFAULT_EXE)
    ap.add_argument("--raw", help="module image dumped with --nui_dump_module_image")
    ap.add_argument("--sidecar", help=".json sidecar of the raw image")
    ap.add_argument("--map", default=milo_sym.DEFAULT_MAP)
    ap.add_argument("--words", type=int, default=24)
    ap.add_argument("--format", choices=["cc", "toml"], default="cc")
    ap.add_argument("--addr", help="hex address instead of a symbol lookup")
    ap.add_argument("--name", help="name to emit when --addr is used")
    ap.add_argument("--verify", help="second image to verify uniqueness in")
    ap.add_argument("--map2", help="map of the second image")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args(argv)

    if args.raw:
        img = RawImage(args.raw, args.sidecar or (os.path.splitext(args.raw)[0] + ".json"))
        syms = []
    else:
        img = milo_dis.Image(args.exe)
        syms = milo_sym.load_map(args.map)
    targets = []
    if args.addr:
        targets.append((args.name or ("sub_%s" % args.addr), int(args.addr, 16)))
    for name in args.names:
        va = find_symbol(syms, name)
        if va is None:
            print("// symbol not found: %s" % name, file=sys.stderr)
            continue
        targets.append((name, va))

    verify_img = milo_dis.Image(args.verify) if args.verify else None
    verify_syms = milo_sym.load_map(args.map2) if args.map2 else None

    for name, va in targets:
        words = read_words(img, va, args.words)
        words, masks = mask_words(words)
        hits = count_matches(img, words, masks)
        status = "unique" if len(hits) == 1 else "%d matches" % len(hits)
        if verify_img is not None:
            vhits = count_matches(verify_img, words, masks)
            expect = find_symbol(verify_syms, name) if verify_syms else None
            vstatus = "%d matches in verify image" % len(vhits)
            if expect is not None:
                vstatus += (", at symbol" if vhits == [expect]
                            else ", expected %08x got %s" % (
                                expect, ["%08x" % h for h in vhits]))
            status += "; " + vstatus
        print("// %s @ %08x: %s" % (name, va, status))
        if args.format == "cc":
            print(emit_cc(name, words, masks, va))
        else:
            print(emit_toml(name, words, masks))


if __name__ == "__main__":
    main(sys.argv[1:])
