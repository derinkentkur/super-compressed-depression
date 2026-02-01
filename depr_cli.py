#!/usr/bin/env python3
# noinspection SpellCheckingInspection

"""
DEPR — Displayable Extreme Picture Repacker (with Compression Map)

What this tool does
-------------------
1) Exhaustive multi-codec compression of a single input image.
2) Displayable container: outputs a custom .depr file that remains directly viewable.
3) Compression Map archival blob (CMAP) for partial/random access decoding.

Input location modes
--------------------
This CLI supports three ways to locate the input file:
  --path <FILE>                 # direct file path
  --find <NAME>                 # search by filename under the user's home (~)
  --find-scope <SCOPE> <NAME>   # search by filename under an explicit scope directory

Output/working locations
------------------------
By default, outputs are placed under "~/depr" (created automatically):
  compress:  ~/depr/<stem>.depr   unless --output is given
  extract:   ~/depr/extracted/<stem>/  unless --outdir is given

CLI
---
  depr compress [--path P | --find NAME | --find-scope SCOPE NAME] [--output OUT.depr] [--epsilon 0.05] [--extreme]
  depr extract  [--path P | --find NAME | --find-scope SCOPE NAME] [--outdir DIR]
  depr inspect  [--path P | --find NAME | --find-scope SCOPE NAME]

Requirements
------------
  - Pillow (PIL)
  - Optional: pillow-avif-plugin (AVIF encode/decode). If unavailable, AVIF is skipped.
  - Optional: brotli (for smallest archival CMAP). If unavailable, CMAP falls back to zlib.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import struct
import sys
# allow fallback
# noinspection PyUnresolvedReferences
import sys as _sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from PIL import Image

DEBUG = False


def _set_mode_interactive(requested: Optional[str]) -> str:
    global DEBUG
    mode_inner_scope = (requested or "").strip().lower()
    if mode_inner_scope not in {"normal", "debug"}:
        try:
            choice = input("Select mode [normal/debug] (default: normal): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "normal"
        mode_inner_scope = choice if choice in {"normal", "debug"} else "normal"
    DEBUG = (mode_inner_scope == "debug")
    return mode_inner_scope

def _dprint(*a, **k):
    if DEBUG:
        print(*a, **k)


# noinspection PyBroadException
try:
    import pillow_avif  # noqa: F401
    AVIF_AVAILABLE = True
except Exception:
    AVIF_AVAILABLE = False

# noinspection PyBroadException
try:
    import brotli
    BROTLI_AVAILABLE = True
except Exception:
    BROTLI_AVAILABLE = False
# noinspection SpellCheckingInspection
MANIFEST_MARKER: bytes = b"\nDEPR\x00MANIFEST\n"
BLOBS_MARKER:    bytes = b"\nDEPR\x00BLOBS\n"
# noinspection SpellCheckingInspection
SUPPORTED_PRIMARY_EXTS = {"jpg", "jpeg", "webp", "avif"}

HOME_BASE = os.path.expanduser("~")
DEFAULT_BASE_DIR = os.path.join(HOME_BASE, "depr")
DEFAULT_EXTRACT_DIR = os.path.join(DEFAULT_BASE_DIR, "extracted")

@dataclass
class Candidate:
    name: str
    ext: str
    mime: str
    kind: str
    settings: Dict[str, object]
    data: bytes
    @property
    def size(self) -> int:
        return len(self.data)

# ---------------- file location helpers ----------------

def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def _search_by_name(scope: str, name: str) -> Optional[str]:
    scope = _expand(scope)
    best: Optional[Tuple[float, str]] = None
    for root, _dirs, files in os.walk(scope):
        for f in files:
            if f == name:
                full = os.path.join(root, f)
                try:
                    m = os.path.getmtime(full)
                except OSError:
                    m = 0.0
                if best is None or m > best[0]:
                    best = (m, full)
    return best[1] if best else None

def resolve_input_path(path: Optional[str], find_name: Optional[str], find_scope: Optional[Tuple[str, str]]) -> str:
    if path:
        p = _expand(path)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return p
    if find_name:
        candidate = _search_by_name(HOME_BASE, find_name)
        if not candidate:
            raise FileNotFoundError(f"'{find_name}' not found under ~")
        return candidate
    if find_scope:
        scope, name = find_scope
        candidate = _search_by_name(scope, name)
        if not candidate:
            raise FileNotFoundError(f"'{name}' not found under scope '{scope}'")
        return candidate
    raise ValueError("one of --path/--find/--find-scope is required")

# ---------------- codec helpers ----------------

def _encode_to_bytes(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()

def _jpeg(img: Image.Image, quality: int) -> Candidate:
    data = _encode_to_bytes(img, "JPEG", quality=max(1, int(quality)), optimize=True, progressive=True)
    return Candidate(name=f"jpeg_q{quality}", ext="jpg", mime="image/jpeg", kind="lossy",
                     settings={"quality": quality, "progressive": True}, data=data)

def _webp_lossy(img: Image.Image, quality: int, method: int = 6) -> Candidate:
    data = _encode_to_bytes(img, "WEBP", quality=max(0, int(quality)), method=int(method))
    return Candidate(name=f"webp_q{quality}", ext="webp", mime="image/webp", kind="lossy",
                     settings={"quality": quality, "method": method}, data=data)

def _webp_lossless(img: Image.Image) -> Candidate:
    data = _encode_to_bytes(img, "WEBP", lossless=True)
    return Candidate(name="webp_lossless", ext="webp", mime="image/webp", kind="lossless",
                     settings={"lossless": True}, data=data)

def _png_opt(img: Image.Image) -> Candidate:
    data = _encode_to_bytes(img, "PNG", optimize=True)
    return Candidate(name="png_opt", ext="png", mime="image/png", kind="lossless",
                     settings={"optimize": True}, data=data)

def _png8(img: Image.Image, colors: int = 256) -> Candidate:
    pal_img = img.convert("P", palette=getattr(Image, "ADAPTIVE", 0), colors=max(2, min(256, int(colors))))
    data = _encode_to_bytes(pal_img, "PNG", optimize=True)
    return Candidate(name=f"png8_{colors}", ext="png", mime="image/png", kind="lossless",
                     settings={"indexed_colors": colors, "optimize": True}, data=data)

def _avif_lossy(img: Image.Image, quality: int = 28) -> Optional[Candidate]:
    if not AVIF_AVAILABLE:
        return None
    data = _encode_to_bytes(img, "AVIF", quality=int(quality))
    return Candidate(name=f"avif_q{quality}", ext="avif", mime="image/avif", kind="lossy",
                     settings={"quality": quality}, data=data)

def _avif_lossless(img: Image.Image) -> Optional[Candidate]:
    if not AVIF_AVAILABLE:
        return None
    data = _encode_to_bytes(img, "AVIF", lossless=True)
    return Candidate(name="avif_lossless", ext="avif", mime="image/avif", kind="lossless",
                     settings={"lossless": True}, data=data)

def generate_candidates(img: Image.Image, extreme: bool = False) -> List[Candidate]:
    # noinspection SpellCheckingInspection
    cands: List[Candidate] = []
    lossy_qualities = [0, 1, 5, 10, 20, 40] if extreme else [60, 40, 30]
    for q in [q for q in lossy_qualities if q <= 60]:
        try:
            cands.append(_jpeg(img, q))
        except (OSError, ValueError, RuntimeError) as e:
            _dprint(f"debug: jpeg q{q} failed: {e}")
    for q in lossy_qualities:
        try:
            cands.append(_webp_lossy(img, q))
        except (OSError, ValueError, RuntimeError) as e:
            _dprint(f"debug: webp q{q} failed: {e}")
    if AVIF_AVAILABLE:
        for q in ([4, 10, 20] if extreme else [28, 20]):
            try:
                av = _avif_lossy(img, q)
                if av:
                    cands.append(av)
            except (OSError, ValueError, RuntimeError) as e:
                _dprint(f"debug: avif q{q} failed: {e}")
    for fn in (_png_opt, _webp_lossless):
        try:
            cands.append(fn(img))
        except (OSError, ValueError, RuntimeError) as e:
            _dprint(f"debug: lossless {fn.__name__} failed: {e}")
    for k in (256, 128, 64):
        try:
            cands.append(_png8(img, colors=k))
        except (OSError, ValueError, RuntimeError) as e:
            _dprint(f"debug: png8 {k} failed: {e}")
    if AVIF_AVAILABLE:
        try:
            avl = _avif_lossless(img)
            if avl:
                cands.append(avl)
        except (OSError, ValueError, RuntimeError) as e:
            _dprint(f"debug: avif lossless failed: {e}")
    _dprint(f"debug: generated {len(cands)} candidates")
    return [c for c in cands if c is not None]

# noinspection SpellCheckingInspection
def pick_primary(cands: List[Candidate]) -> Candidate:
    displayables = [c for c in cands if c.ext in SUPPORTED_PRIMARY_EXTS]
    return min(displayables or cands, key=lambda c: c.size)

# noinspection SpellCheckingInspection
def choose_small_set(cands: List[Candidate], epsilon: float = 0.05) -> Tuple[int, List[Candidate]]:
    min_size = min(c.size for c in cands)
    threshold = int(min_size * (1.0 + max(0.0, float(epsilon))))
    near = [c for c in cands if c.size <= threshold]
    out: Dict[Tuple[str, str], Candidate] = {}
    for c in sorted(near, key=lambda x: x.size):
        out.setdefault((c.name, c.ext), c)
    _dprint(f"debug: min={min_size} threshold={threshold} kept={len(out)} of {len(cands)}")
    return min_size, list(out.values())

# ---------------- compression map ----------------

def _quantize_indexed(img: Image.Image, max_colors: int = 256) -> Tuple[Image.Image, bytes]:
    if img.mode != "P":
        pal_img = img.convert("P", palette=getattr(Image, "ADAPTIVE", 0), colors=max(2, min(256, int(max_colors))))
    else:
        pal_img = img
    pal = pal_img.getpalette() or []
    pal_bytes = bytes(pal[: 256 * 3]).ljust(256 * 3, b"\x00")
    return pal_img, pal_bytes

def make_cmap_candidate(img: Image.Image, rows_per_chunk: int = 64, codec: str = "zlib") -> Candidate:
    idx_img, palette_bytes = _quantize_indexed(img, max_colors=256)
    w, h = idx_img.size
    idx_bytes = idx_img.tobytes()
    def _compress(block_inner_scope: bytes) -> bytes:
        if codec == "brotli" and BROTLI_AVAILABLE:
            return brotli.compress(block_inner_scope)
        import zlib
        co = zlib.compressobj(level=9)
        return co.compress(block_inner_scope) + co.flush()
    offsets: List[int] = []
    sizes: List[int] = []
    payload_buf = io.BytesIO()
    cumulative = 0
    for row in range(0, h, rows_per_chunk):
        end = min(h, row + rows_per_chunk)
        start = row * w
        block = idx_bytes[start : start + (end - row) * w]
        comp = _compress(block)
        payload_buf.write(comp)
        offsets.append(cumulative)
        sizes.append(len(comp))
        cumulative += len(comp)
    payload = payload_buf.getvalue()
    cmap = {
        "type": "row-chunks",
        "codec": ("brotli" if (codec == "brotli" and BROTLI_AVAILABLE) else "zlib"),
        "image": {"width": w, "height": h, "storage": "indexed8", "rows_per_chunk": rows_per_chunk},
        "palette_bytes": len(palette_bytes),
        "chunks": {"count": len(offsets), "offsets": offsets, "sizes": sizes, "cumulative_end": [o + s for o, s in zip(offsets, sizes)]},
    }
    # noinspection SpellCheckingInspection
    header = b"DEPRCMAP\x00v1\n" + json.dumps({"w": w, "h": h, "r": rows_per_chunk, "codec": cmap["codec"], "pal": len(palette_bytes)}, separators=(",", ":")).encode("utf-8") + b"\n"
    blob = header + palette_bytes + payload
    return Candidate(name=f"cmap_{cmap['codec']}_r{rows_per_chunk}", ext="cmap", mime="application/octet-stream", kind="archive", settings={"compression_map": cmap}, data=blob)

# ---------------- container I/O ----------------

def write_depr(primary: Candidate, others: List[Candidate], out_path: str, source_name: str, compression_map: Optional[dict] = None) -> None:
    manifest = {
        "version": 2,
        "original": os.path.basename(source_name),
        "primary": {"name": primary.name, "ext": primary.ext, "mime": primary.mime, "kind": primary.kind, "settings": primary.settings, "size": primary.size},
        "alternatives": [{"name": c.name, "ext": c.ext, "mime": c.mime, "kind": c.kind, "settings": c.settings, "size": c.size} for c in others],
        "note": "File starts with primary image bytes for displayability; trailing manifest/blobs follow markers.",
    }
    if compression_map is not None:
        manifest["compression_map"] = compression_map
    dirpath = os.path.dirname(_expand(out_path))
    _ensure_dir(dirpath)
    with open(out_path, "wb") as f:
        f.write(primary.data)
        f.write(MANIFEST_MARKER)
        f.write(json.dumps(manifest, separators=(",", ":")).encode("utf-8"))
        f.write(BLOBS_MARKER)
        def _u32(n: int) -> bytes: return struct.pack(">I", int(n))
        def _u64(n: int) -> bytes: return struct.pack(">Q", int(n))
        f.write(_u32(len(others)))
        for c in others:
            name_b = c.name.encode("utf-8"); ext_b = c.ext.encode("utf-8")
            f.write(_u32(len(name_b))); f.write(name_b)
            f.write(_u32(len(ext_b)));  f.write(ext_b)
            f.write(_u64(len(c.data)))
            f.write(c.data)

def read_manifest(path: str) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        data = f.read()
    # noinspection SpellCheckingInspection
    mpos = data.find(MANIFEST_MARKER)
    if mpos < 0:
        raise ValueError("Not a DEPR file: manifest marker not found")
    # noinspection SpellCheckingInspection
    jstart = mpos + len(MANIFEST_MARKER)
    # noinspection SpellCheckingInspection
    bpos = data.find(BLOBS_MARKER, jstart)
    if bpos < 0:
        raise ValueError("Corrupt DEPR: blobs marker not found")
    manifest_json = data[jstart:bpos]
    manifest = json.loads(manifest_json.decode("utf-8"))
    return manifest, bpos + len(BLOBS_MARKER)

def extract(path: str, outdir: str) -> None:
    _ensure_dir(outdir)
    with open(path, "rb") as f:
        data = f.read()
    # noinspection SpellCheckingInspection
    mpos = data.find(MANIFEST_MARKER)
    if mpos < 0:
        raise ValueError("Not a DEPR file: manifest marker not found")
    primary_bytes = data[:mpos]
    manifest, blobs_off = read_manifest(path)
    p = manifest["primary"]
    primary_name = f"primary.{p['ext']}"
    with open(os.path.join(outdir, primary_name), "wb") as fp:
        fp.write(primary_bytes)
    buf = memoryview(data)[blobs_off:]

    def _read_u32(off_inner_scope: int) -> Tuple[int, int]:
        """Read a 4-byte unsigned integer from the buffer."""
        val = struct.unpack(">I", buf[off_inner_scope:off_inner_scope + 4])[0]
        return val, off_inner_scope + 4

    def _read_u64(off_inner_scope: int) -> Tuple[int, int]:
        """Read an 8-byte unsigned integer from the buffer."""
        val = struct.unpack(">Q", buf[off_inner_scope:off_inner_scope + 8])[0]
        return val, off_inner_scope + 8

    count, off = _read_u32(0)

    for _ in range(count):
        # noinspection SpellCheckingInspection
        nlen, off = _read_u32(off); name = bytes(buf[off:off+nlen]).decode("utf-8"); off += nlen
        elen, off = _read_u32(off); ext = bytes(buf[off:off+elen]).decode("utf-8"); off += elen
        size, off = _read_u64(off); blob = bytes(buf[off:off+size]); off += size
        # noinspection SpellCheckingInspection
        fname = f"alt_{name}.{ext}"
        with open(os.path.join(outdir, fname), "wb") as fo:
            fo.write(blob)
    if "compression_map" in manifest:
        with open(os.path.join(outdir, "compression_map.json"), "w", encoding="utf-8") as cmf:
            json.dump(manifest["compression_map"], cmf, indent=2)

# ---------------- orchestration ----------------

def compress(input_path: str, output_path: str, epsilon: float = 0.05, extreme: bool = False) -> None:
    _dprint(f"debug: compress input={input_path} output={output_path} eps={epsilon} extreme={extreme}")
    img = Image.open(input_path)
    if img.mode not in ("RGB", "RGBA", "P"):
        img = img.convert("RGB")
    # noinspection SpellCheckingInspection
    cands = generate_candidates(img, extreme=extreme)
    if not cands:
        raise RuntimeError("No candidates produced — check encoder availability.")
    # noinspection SpellCheckingInspection
    cmap_cands: List[Candidate] = []
    try:
        cmap_cands.append(make_cmap_candidate(img, rows_per_chunk=64, codec="zlib"))
        if BROTLI_AVAILABLE:
            cmap_cands.append(make_cmap_candidate(img, rows_per_chunk=64, codec="brotli"))
    except (OSError, ValueError, RuntimeError) as e:
        _dprint(f"debug: cmap build failed: {e}")
    all_for_choice = cands + cmap_cands
    min_size, small_set = choose_small_set(all_for_choice, epsilon=epsilon)
    primary = pick_primary(small_set)
    others = [c for c in small_set if c is not primary]
    compression_map: Optional[dict] = None
    if cmap_cands:
        best_cmap = min(cmap_cands, key=lambda c: c.size)
        if best_cmap not in others and best_cmap is not primary:
            others.append(best_cmap)
        compression_map = cast(Optional[dict], best_cmap.settings.get("compression_map")) if "compression_map" in best_cmap.settings else None

    write_depr(primary, others, output_path, source_name=input_path, compression_map=compression_map)
    print("== DEPR Report ==")
    print(f"Input: {input_path}")
    print(f"Total candidates: {len(all_for_choice)}; Small-set (<= {(1+epsilon)*100:.1f}% of min): {len(small_set)}")
    print(f"Minimum size overall: {min_size} bytes")
    print(f"Primary: {primary.name} ({primary.size} bytes) -> {output_path}")
    if DEBUG:
        for c in sorted(all_for_choice, key=lambda x: x.size)[:10]:
            print(f"debug: cand {c.name}.{c.ext} {c.size} B kind={c.kind}")
    if others:
        print("Alternatives included:")
        for c in sorted([x for x in others if x is not primary], key=lambda x: x.size):
            print(f"  - {c.name:20s} {c.size} B")

# ---------------- CLI + new location modes ----------------

def _add_location_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", dest="in_path", help="Direct path to input file")
    g.add_argument("--find", dest="in_find", help="Find by filename under ~")
    g.add_argument("--find-scope", dest="in_find_scope", nargs=2, metavar=("SCOPE", "NAME"), help="Find by filename under SCOPE directory")

def _resolve_and_defaults_for_compress(args: argparse.Namespace) -> Tuple[str, str]:
    in_file = resolve_input_path(args.in_path, args.in_find, tuple(args.in_find_scope) if args.in_find_scope else None)
    if args.output:
        out_file = _expand(args.output)
    else:
        _ensure_dir(DEFAULT_BASE_DIR)
        stem = os.path.splitext(os.path.basename(in_file))[0]
        out_file = os.path.join(DEFAULT_BASE_DIR, f"{stem}.depr")
    return in_file, out_file

def _resolve_and_defaults_for_extract_or_inspect(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    in_file = resolve_input_path(args.in_path, args.in_find, tuple(args.in_find_scope) if args.in_find_scope else None)
    # noinspection PyUnusedLocal
    out_dir = None
    if getattr(args, "outdir", None):
        out_dir = _expand(args.outdir)
    else:
        stem = os.path.splitext(os.path.basename(in_file))[0]
        out_dir = os.path.join(DEFAULT_EXTRACT_DIR, stem)
    _ensure_dir(out_dir)
    return in_file, out_dir

def cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="depr", description="Displayable Extreme Picture Repacker (.depr)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="Compress an image into a .depr container")
    _add_location_args(pc)
    pc.add_argument("--output", help="Output .depr file path (default: ~/depr/<stem>.depr)")
    pc.add_argument("--epsilon", type=float, default=0.05, help="Include candidates within (1+epsilon)*min size [default: 0.05]")
    pc.add_argument("--extreme", action="store_true", help="Use ultra-low quality settings")
    pc.add_argument("--mode", choices=["normal", "debug"], help="Execution mode")

    pe = sub.add_parser("extract", help="Extract primary & alternatives from a .depr file")
    _add_location_args(pe)
    pe.add_argument("--outdir", help="Directory to write extracted files (default: ~/depr/extracted/<stem>)")
    pe.add_argument("--mode", choices=["normal", "debug"], help="Execution mode")

    pi = sub.add_parser("inspect", help="Show .depr manifest")
    _add_location_args(pi)
    pi.add_argument("--mode", choices=["normal", "debug"], help="Execution mode")

    args = p.parse_args(argv)

    _set_mode_interactive(getattr(args, "mode", None))

    try:
        if args.cmd == "compress":
            in_file, out_file = _resolve_and_defaults_for_compress(args)
            compress(in_file, out_file, epsilon=args.epsilon, extreme=args.extreme)
            return 0
        elif args.cmd == "extract":
            in_file, out_dir = _resolve_and_defaults_for_extract_or_inspect(args)
            extract(in_file, out_dir)
            return 0
        elif args.cmd == "inspect":
            in_file, _ = _resolve_and_defaults_for_extract_or_inspect(args)
            manifest, _ = read_manifest(in_file)
            print(json.dumps(manifest, indent=2))
            return 0
        else:
            print("unknown command")
            return 2
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 66
    except Exception as e:
        print(f"error: {e}")
        return 1

# ---------------- Debug REPL (interactive) ----------------

CONFIG = {
    "epsilon": 0.05,
    "extreme": False,
    "base_dir": DEFAULT_BASE_DIR,
    "extract_base": DEFAULT_EXTRACT_DIR,
}

import shlex

def _to_bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "on"}

def _debug_help() -> None:
    print("""
DEPR debug console
Commands:
  help                                Show this help
  show                                Show current CONFIG
  set <key> <value>                   Set a config value (epsilon|extreme|base_dir|extract_base)
  get <key>                           Get a config value
  compress [--path P | --find N | --find-scope S N] [--output OUT]
                                      Run compress with interactive defaults
  extract  [--path P | --find N | --find-scope S N] [--outdir DIR]
                                      Run extract
  inspect  [--path P | --find N | --find-scope S N]
                                      Show manifest
  quit | exit                         Leave debug console
""".strip())


def _resolve_input_from_tokens(tokens: List[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    in_path = None; in_find = None; in_scope = None; in_name = None
    it = iter(tokens)
    for t in it:
        if t == "--path":
            in_path = next(it, None)
        elif t == "--find":
            in_find = next(it, None)
        elif t == "--find-scope":
            in_scope = next(it, None)
            in_name = next(it, None)
        elif t == "--":
            break
    return in_path, in_find, (in_scope if in_scope else None), (in_name if in_name else None)


# noinspection PyBroadException
def _debug_repl() -> int:
    _debug_help()
    while True:
        try:
            line = input("depr> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"quit", "exit"}:
            return 0
        if line == "help":
            _debug_help(); continue
        if line == "show":
            print(json.dumps(CONFIG, indent=2)); continue
        parts = shlex.split(line)
        cmd, *args = parts
        if cmd == "set" and len(args) >= 2:
            key, value = args[0], " ".join(args[1:])
            if key == "epsilon":
                try: CONFIG["epsilon"] = float(value)
                except ValueError: print("invalid float for epsilon")
            elif key == "extreme":
                CONFIG["extreme"] = _to_bool(value)
            elif key == "base_dir":
                CONFIG["base_dir"] = _expand(value)
            elif key == "extract_base":
                CONFIG["extract_base"] = _expand(value)
            else:
                print("unknown key")
            continue
        if cmd == "get" and len(args) == 1:
            key = args[0]
            print(CONFIG.get(key, None))
            continue
# nyan_cat
        if cmd == "nyan_cat":
            print("""
        •.,¸,.•*`•.,¸¸,.•*¯ ╭━━━━━━━━━━╮
        •.,¸,.•*¯`•.,¸,.•*¯.|:::::/\\___/\\
        •.,¸,.•*¯`•.,¸,.•* <|::: :(｡ ●ω●｡)
        •.,¸,.•¯•.,¸,.•╰ * し---し---   Ｊ
                    """)
        if cmd in {"compress", "extract", "inspect"}:
            in_path, in_find, in_scope, in_name = _resolve_input_from_tokens(args)
            try:
                in_file = resolve_input_path(in_path, in_find, (in_scope, in_name) if (in_scope and in_name) else None)
            except Exception as e:
                print(f"error: {e}"); continue
            if cmd == "compress":
                out = None
                if "--output" in args:
                    try:
                        out = args[args.index("--output")+1]
                    except (ValueError, IndexError): out = None
                if not out:
                    _ensure_dir(CONFIG["base_dir"])
                    stem = os.path.splitext(os.path.basename(in_file))[0]
                    out = os.path.join(CONFIG["base_dir"], f"{stem}.depr")
                try:
                    compress(in_file, out, epsilon=CONFIG["epsilon"], extreme=CONFIG["extreme"])
                except Exception as e:
                    print(f"error: {e}")
            elif cmd == "extract":
                outdir = None
                if "--outdir" in args:
                    try:
                        outdir = args[args.index("--outdir")+1]
                    except: outdir = None
                if not outdir:
                    stem = os.path.splitext(os.path.basename(in_file))[0]
                    outdir = os.path.join(CONFIG["extract_base"], stem)
                try:
                    extract(in_file, _expand(outdir))
                except Exception as e:
                    print(f"error: {e}")
            else:
                try:
                    manifest, _ = read_manifest(in_file)
                    print(json.dumps(manifest, indent=2))
                except Exception as e:
                    print(f"error: {e}")
            continue
        print("unknown command; type 'help')")
    return 0
# ---------------- Interactive wizard (normal mode, no args) ------------------

def _prompt_nonempty(prompt: str, default: Optional[str] = None) -> str:
    while True:
        val = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
        if not val and default is not None:
            return default
        if val:
            return val


def _wizard_normal() -> int:
    print("Entering interactive setup (no/invalid arguments detected).")
    cmd = _prompt_nonempty("Command (compress/extract/inspect)", "compress").lower()
    if cmd not in {"compress", "extract", "inspect"}:
        print("unknown command"); return 2
    mode_inner_scope = _prompt_nonempty("Input mode (path/find/find-scope)", "path").lower()
    in_path = in_find = scope = name = None
    if mode_inner_scope == "path":
        in_path = _prompt_nonempty("Path to input file")
    elif mode_inner_scope == "find":
        in_find = _prompt_nonempty("Filename to find under ~")
    elif mode_inner_scope == "find-scope":
        scope = _prompt_nonempty("Scope directory (can be /, ., ~, etc.)")
        name = _prompt_nonempty("Filename to find in scope")
    else:
        print("unknown input mode"); return 2

    try:
        in_file = resolve_input_path(in_path, in_find, (scope, name) if (scope and name) else None)
    except Exception as e:
        print(f"error: {e}"); return 66

    if cmd == "compress":
        epsilon_s = _prompt_nonempty("Epsilon (0..1)", str(CONFIG.get("epsilon", 0.05)))
        try:
            eps = float(epsilon_s)
        except ValueError:
            eps = 0.05
        extreme_ans = _prompt_nonempty("Extreme mode (true/false)", "false").lower()
        extreme = extreme_ans in {"1", "true", "yes", "on"}
        stem = os.path.splitext(os.path.basename(in_file))[0]
        _ensure_dir(CONFIG["base_dir"])
        out = _prompt_nonempty("Output .depr file", os.path.join(CONFIG["base_dir"], f"{stem}.depr"))
        try:
            compress(in_file, out, epsilon=eps, extreme=extreme)
            return 0
        except Exception as e:
            print(f"error: {e}"); return 1

    elif cmd == "extract":
        stem = os.path.splitext(os.path.basename(in_file))[0]
        _ensure_dir(CONFIG["extract_base"])
        outdir = _prompt_nonempty("Output directory", os.path.join(CONFIG["extract_base"], stem))
        try:
            extract(in_file, outdir)
            return 0
        except Exception as e:
            print(f"error: {e}"); return 1

    else:
        try:
            manifest, _ = read_manifest(in_file)
            print(json.dumps(manifest, indent=2))
            return 0
        except Exception as e:
            print(f"error: {e}"); return 1


if __name__ == "__main__":
    mode = _set_mode_interactive(None)
    if mode == "debug":
        rc = _debug_repl()
        sys.exit(rc)
    if len(sys.argv) <= 1:
        sys.exit(_wizard_normal())
    try:
        sys.exit(cli())
    except SystemExit as se:
        code = getattr(se, "code", 0)
        if code == 2:
            sys.exit(_wizard_normal())
        else:
            raise
