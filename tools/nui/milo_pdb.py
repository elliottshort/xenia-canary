#!/usr/bin/env python3
"""Extract function/data symbols from a Microsoft PDB 7.0 (MSF) file.

Pure Python -- no third-party PDB library required.  pefile is used only as a
fallback source of PE section headers (the DBI "section headers" debug stream is
preferred when present) and for the image base.

The point of this script is to recover the STATIC (module-local, S_LPROC32)
functions that a linker .map file does not list.

Usage:
  python milo_pdb.py [PDB] [EXE]        # defaults to the Project Milo files
Output:
  milo_symbols.tsv  next to this script:  VA_hex \t size_hex \t kind \t name \t module
"""

import mmap
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MILO_DIR = r"C:\Users\ellio\Documents\github\xenia-canary\xbe\Project Milo (May 17 2010)"
DEFAULT_PDB = os.path.join(MILO_DIR, "miloReleaseLIB.pdb")
DEFAULT_EXE = os.path.join(MILO_DIR, "miloReleaseLIB.exe")
DEFAULT_TSV = os.path.join(HERE, "milo_symbols.tsv")

MSF_MAGIC = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\x00\x00\x00"

# ---------------------------------------------------------------------------
# MSF container
# ---------------------------------------------------------------------------


class MSF(object):
    """Multi-Stream Format (PDB 7.0) container."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:len(MSF_MAGIC)] != MSF_MAGIC:
            raise ValueError("not a PDB 7.0 (MSF) file: bad magic")
        off = len(MSF_MAGIC)
        (self.block_size, self.free_block_map, self.num_blocks,
         self.num_dir_bytes, self._unknown, self.block_map_addr) = \
            struct.unpack_from("<6I", self.mm, off)
        if self.block_size not in (512, 1024, 2048, 4096, 8192):
            raise ValueError("implausible block size %d" % self.block_size)
        self.streams = self._read_directory()

    # -- low level ----------------------------------------------------------
    def block(self, idx):
        if idx >= self.num_blocks:
            raise ValueError("block %d out of range" % idx)
        o = idx * self.block_size
        return self.mm[o:o + self.block_size]

    def _blocks(self, indices, size):
        out = bytearray()
        for b in indices:
            out += self.block(b)
            if len(out) >= size:
                break
        return bytes(out[:size])

    def _read_directory(self):
        bs = self.block_size
        n_dir_blocks = (self.num_dir_bytes + bs - 1) // bs
        # The block-map itself is a contiguous run of blocks at block_map_addr.
        map_bytes_needed = n_dir_blocks * 4
        n_map_blocks = (map_bytes_needed + bs - 1) // bs
        raw_map = self._blocks(range(self.block_map_addr,
                                     self.block_map_addr + n_map_blocks),
                               map_bytes_needed)
        dir_blocks = struct.unpack("<%dI" % n_dir_blocks, raw_map)
        d = self._blocks(dir_blocks, self.num_dir_bytes)

        pos = 0
        (num_streams,) = struct.unpack_from("<I", d, pos)
        pos += 4
        sizes = list(struct.unpack_from("<%dI" % num_streams, d, pos))
        pos += 4 * num_streams
        streams = []
        for sz in sizes:
            if sz == 0xFFFFFFFF:  # nil stream
                sz = 0
            nb = (sz + bs - 1) // bs
            blocks = struct.unpack_from("<%dI" % nb, d, pos) if nb else ()
            pos += 4 * nb
            streams.append((sz, blocks))
        return streams

    # -- public -------------------------------------------------------------
    @property
    def num_streams(self):
        return len(self.streams)

    def stream_size(self, idx):
        if idx is None or idx == 0xFFFF or idx >= len(self.streams):
            return 0
        return self.streams[idx][0]

    def read_stream(self, idx):
        if idx is None or idx == 0xFFFF or idx >= len(self.streams):
            return b""
        size, blocks = self.streams[idx]
        if size == 0:
            return b""
        return self._blocks(blocks, size)


# ---------------------------------------------------------------------------
# CodeView symbol records
# ---------------------------------------------------------------------------

S_THUNK32 = 0x1102
S_LDATA32 = 0x110C
S_GDATA32 = 0x110D
S_PUB32 = 0x110E

# NOTE: the CodeView numbering is S_LPROC32 == 0x110F, S_GPROC32 == 0x1110 and
# S_REGREL32 == 0x1111.  0x1111 must NOT be treated as a proc record -- doing so
# mis-parses the ~33k register-relative local records in this PDB.  Verified
# against the reference records in the globals stream: the 0x1110 count matches
# S_PROCREF (0x1125) exactly and the 0x110F count matches S_LPROCREF (0x1127).
PROC_KINDS = {
    0x110F: "LPROC", 0x1110: "GPROC",
    0x1146: "LPROC_ID", 0x1147: "GPROC_ID",
    0x1155: "LPROC_DPC", 0x1156: "LPROC_DPC_ID",
}
# S_LDATA32/S_GDATA32 and their thread-local twins share one layout.
DATA_KINDS = {0x110C: "LDATA", 0x110D: "GDATA",
              0x1112: "LTHREAD", 0x1113: "GTHREAD"}

KIND_PRIORITY = {"GPROC": 5, "LPROC": 5, "GPROC_ID": 5, "LPROC_ID": 5,
                 "LPROC_DPC": 5, "LPROC_DPC_ID": 5,
                 "THUNK": 4, "GDATA": 3, "LDATA": 3,
                 "GTHREAD": 3, "LTHREAD": 3, "PUB": 1}


def _cstr(buf, off, end):
    z = buf.find(b"\x00", off, end)
    if z < 0:
        z = end
    return buf[off:z].decode("utf-8", "replace")


class SymbolSink(object):
    def __init__(self):
        self.by_va = {}
        self.counts = {}
        self.skipped_kinds = {}
        self.malformed = 0

    def add(self, va, size, kind, name, module):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        prev = self.by_va.get(va)
        if prev is not None:
            if KIND_PRIORITY.get(kind, 0) <= KIND_PRIORITY.get(prev[1], 0):
                return
        self.by_va[va] = (size, kind, name, module)

    def note_skip(self, kind):
        self.skipped_kinds[kind] = self.skipped_kinds.get(kind, 0) + 1


def parse_symbols(buf, start, end, sink, module, sect_va, image_base):
    """Iterate CodeView symbol records in buf[start:end]."""
    pos = start
    while pos + 4 <= end:
        try:
            reclen, kind = struct.unpack_from("<HH", buf, pos)
        except struct.error:
            sink.malformed += 1
            break
        if reclen < 2:
            sink.malformed += 1
            break
        nxt = pos + 2 + reclen           # length field excludes itself
        if nxt > end:
            sink.malformed += 1
            break
        body = pos + 4                   # after length + kind
        try:
            if kind in PROC_KINDS:
                # parent end next len dbgstart dbgend typeindex offset seg flags name
                if body + 35 <= nxt:
                    (_par, _end, _nxt, plen, _ds, _de, _ti, off) = \
                        struct.unpack_from("<8I", buf, body)
                    (seg,) = struct.unpack_from("<H", buf, body + 32)
                    name = _cstr(buf, body + 35, nxt)
                    va = sect_va(seg, off, image_base)
                    if va is not None and name:
                        sink.add(va, plen, PROC_KINDS[kind], name, module)
                else:
                    sink.malformed += 1
            elif kind in DATA_KINDS:
                if body + 10 <= nxt:
                    (_ti, off) = struct.unpack_from("<2I", buf, body)
                    (seg,) = struct.unpack_from("<H", buf, body + 8)
                    name = _cstr(buf, body + 10, nxt)
                    va = sect_va(seg, off, image_base)
                    if va is not None and name:
                        sink.add(va, 0, DATA_KINDS[kind], name, module)
                else:
                    sink.malformed += 1
            elif kind == S_PUB32:
                if body + 10 <= nxt:
                    (_flags, off) = struct.unpack_from("<2I", buf, body)
                    (seg,) = struct.unpack_from("<H", buf, body + 8)
                    name = _cstr(buf, body + 10, nxt)
                    va = sect_va(seg, off, image_base)
                    if va is not None and name:
                        sink.add(va, 0, "PUB", name, module)
                else:
                    sink.malformed += 1
            elif kind == S_THUNK32:
                # parent(4) end(4) next(4) off(4) seg(2) len(2) ord(1) name
                if body + 21 <= nxt:
                    (_par, _end, _nxt, off) = struct.unpack_from("<4I", buf, body)
                    (seg, tlen) = struct.unpack_from("<2H", buf, body + 16)
                    name = _cstr(buf, body + 21, nxt)
                    va = sect_va(seg, off, image_base)
                    if va is not None and name:
                        sink.add(va, tlen, "THUNK", name, module)
                else:
                    sink.malformed += 1
            else:
                sink.note_skip(kind)
        except (struct.error, IndexError, UnicodeError, ValueError):
            sink.malformed += 1
        pos = nxt
    return pos


# ---------------------------------------------------------------------------
# DBI stream
# ---------------------------------------------------------------------------

DBI_STREAM = 3
DBI_KNOWN_VERSIONS = (930803, 19960307, 19970606, 19990903, 20009044)
DBG_SECTION_HDR = 5
DBG_SECTION_HDR_ORIG = 10


class DBI(object):
    def __init__(self, data):
        if len(data) < 64:
            raise ValueError("DBI stream too small (%d bytes)" % len(data))
        (self.sig, self.version, self.age, self.gs_stream, self.build,
         self.ps_stream, self.pdb_dll_ver, self.sym_record_stream,
         self.pdb_dll_rbld, self.modinfo_size, self.seccontrib_size,
         self.secmap_size, self.srcinfo_size, self.typeserver_map_size,
         self.mfc_typeserver_index, self.dbg_header_size, self.ec_size,
         self.flags, self.machine, self._pad) = \
            struct.unpack_from("<iIIHHHHHHiiiiiIiiHHI", data, 0)
        if self.sig != -1:
            raise ValueError("unexpected DBI signature 0x%08x (want 0xffffffff)"
                             % (self.sig & 0xFFFFFFFF))
        if self.version not in DBI_KNOWN_VERSIONS:
            raise ValueError("unknown DBI version %d" % self.version)
        self.data = data
        o = 64
        self.modinfo = (o, o + self.modinfo_size)
        o += self.modinfo_size
        self.seccontrib = (o, o + self.seccontrib_size)
        o += self.seccontrib_size
        self.secmap = (o, o + self.secmap_size)
        o += self.secmap_size
        self.srcinfo = (o, o + self.srcinfo_size)
        o += self.srcinfo_size
        self.typeservermap = (o, o + self.typeserver_map_size)
        o += self.typeserver_map_size
        self.ec = (o, o + self.ec_size)
        o += self.ec_size
        self.dbg_header = (o, min(o + self.dbg_header_size, len(data)))

    def modules(self):
        """Yield (sym_stream_index, sym_byte_size, module_name, obj_name)."""
        d = self.data
        pos, end = self.modinfo
        while pos + 64 <= end:
            _flags, sym_stream = struct.unpack_from("<HH", d, pos + 32)
            sym_size, _c11, _c13 = struct.unpack_from("<3I", d, pos + 36)
            name_off = pos + 64
            mod_name = _cstr(d, name_off, end)
            o2 = d.find(b"\x00", name_off, end)
            if o2 < 0:
                break
            o2 += 1
            obj_name = _cstr(d, o2, end)
            nxt = d.find(b"\x00", o2, end)
            if nxt < 0:
                break
            nxt = (nxt + 1 + 3) & ~3
            if nxt <= pos:
                break
            yield sym_stream, sym_size, mod_name, obj_name
            pos = nxt

    def debug_stream(self, which):
        pos, end = self.dbg_header
        if pos + (which + 1) * 2 > end:
            return None
        (idx,) = struct.unpack_from("<H", self.data, pos + which * 2)
        return None if idx == 0xFFFF else idx


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------


def sections_from_stream(raw):
    """Parse an IMAGE_SECTION_HEADER array (40 bytes each)."""
    out = []
    for o in range(0, len(raw) - 39, 40):
        name = raw[o:o + 8].rstrip(b"\x00").decode("ascii", "replace")
        vsize, vaddr = struct.unpack_from("<II", raw, o + 8)
        out.append((name, vaddr, vsize))
    return out


def sections_from_pe(path):
    try:
        import pefile
    except ImportError:
        return None, None
    try:
        pe = pefile.PE(path, fast_load=True)
    except Exception:
        return None, None
    base = pe.OPTIONAL_HEADER.ImageBase
    out = []
    for s in pe.sections:
        out.append((s.Name.rstrip(b"\x00").decode("ascii", "replace"),
                    s.VirtualAddress, s.Misc_VirtualSize))
    pe.close()
    return out, base


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def short_module(mod_name, obj_name):
    m = os.path.basename(mod_name.replace("/", "\\").rstrip("\\"))
    o = os.path.basename(obj_name.replace("/", "\\").rstrip("\\"))
    if o and m and o.lower() != m.lower():
        return "%s:%s" % (o, m)
    return m or o or "?"


def main(argv):
    pdb_path = argv[0] if len(argv) > 0 else DEFAULT_PDB
    exe_path = argv[1] if len(argv) > 1 else DEFAULT_EXE
    out_path = DEFAULT_TSV

    msf = MSF(pdb_path)
    print("MSF: block=%d blocks=%d streams=%d dir=%d bytes" %
          (msf.block_size, msf.num_blocks, msf.num_streams, msf.num_dir_bytes))

    dbi = DBI(msf.read_stream(DBI_STREAM))
    print("DBI: version=%d age=%d machine=0x%04x sym_record_stream=%d "
          "globals=%d publics=%d" %
          (dbi.version, dbi.age, dbi.machine, dbi.sym_record_stream,
           dbi.gs_stream, dbi.ps_stream))

    # --- section headers ---------------------------------------------------
    sections = None
    src = None
    for which in (DBG_SECTION_HDR, DBG_SECTION_HDR_ORIG):
        idx = dbi.debug_stream(which)
        if idx is not None:
            s = sections_from_stream(msf.read_stream(idx))
            if s:
                sections, src = s, "DBI debug header stream %d" % idx
                break
    pe_sections, image_base = sections_from_pe(exe_path)
    if image_base is None:
        image_base = 0x82000000
    if sections is None:
        sections, src = pe_sections, "PE section table"
    if not sections:
        raise SystemExit("no section headers available")
    if pe_sections and len(pe_sections) == len(sections):
        mismatch = [i for i in range(len(sections))
                    if sections[i][1] != pe_sections[i][1]]
        if mismatch:
            print("WARNING: PDB/PE section RVA mismatch at indices %r" % mismatch)
    print("sections: %d from %s, image base 0x%08x" %
          (len(sections), src, image_base))

    sect_rva = [s[1] for s in sections]

    def sect_va(seg, off, base):
        if seg == 0 or seg > len(sect_rva):
            return None
        return base + sect_rva[seg - 1] + off

    sink = SymbolSink()

    # --- publics / globals from the symbol record stream -------------------
    rec = msf.read_stream(dbi.sym_record_stream)
    print("symbol record stream %d: %d bytes" %
          (dbi.sym_record_stream, len(rec)))
    parse_symbols(rec, 0, len(rec), sink, "<publics>", sect_va, image_base)
    del rec
    n_pub = len(sink.by_va)
    print("  -> %d addresses from publics/globals" % n_pub)

    # --- per-module symbols (this is where the statics live) ---------------
    nmods = 0
    nmods_with_syms = 0
    bad_sig = 0
    for sym_stream, sym_size, mod_name, obj_name in dbi.modules():
        nmods += 1
        if sym_stream == 0xFFFF or sym_size < 4:
            continue
        data = msf.read_stream(sym_stream)
        if not data:
            continue
        limit = min(sym_size, len(data))
        (sig,) = struct.unpack_from("<I", data, 0)
        start = 4
        if sig != 4:  # CV_SIGNATURE_C13
            bad_sig += 1
            if sig > 4:
                start = 0  # no signature word at all -- parse from offset 0
        nmods_with_syms += 1
        parse_symbols(data, start, limit, sink,
                      short_module(mod_name, obj_name), sect_va, image_base)

    print("modules: %d total, %d with symbol streams (%d unexpected signatures)"
          % (nmods, nmods_with_syms, bad_sig))

    # --- write ------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for va in sorted(sink.by_va):
            size, kind, name, module = sink.by_va[va]
            f.write("%08x\t%x\t%s\t%s\t%s\n" % (va, size, kind, name, module))

    print("")
    print("wrote %s: %d unique addresses" % (out_path, len(sink.by_va)))
    print("record counts (before de-dup):")
    for k in sorted(sink.counts, key=lambda x: -sink.counts[x]):
        print("  %-12s %d" % (k, sink.counts[k]))
    kept = {}
    for _va, (size, kind, name, module) in sink.by_va.items():
        kept[kind] = kept.get(kind, 0) + 1
    print("kept per kind (after de-dup):")
    for k in sorted(kept, key=lambda x: -kept[x]):
        print("  %-12s %d" % (k, kept[k]))
    print("malformed/truncated records skipped: %d" % sink.malformed)
    top = sorted(sink.skipped_kinds.items(), key=lambda kv: -kv[1])
    print("unhandled record kinds: %d distinct, %d records" %
          (len(top), sum(sink.skipped_kinds.values())))
    for k, n in top[:24]:
        print("  0x%04x  %d" % (k, n))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
