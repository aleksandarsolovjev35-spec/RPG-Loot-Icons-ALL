#!/usr/bin/env python3
"""Собрать Foundry VTT модуль из набора иконок.

Берёт готовый набор (по умолчанию icons-256) и упаковывает его в
дистрибутив модуля Foundry: module.json + icons/ + packs/ + manifest.json.

    python3 tools/build_foundry_module.py --set icons-256 --out dist/rpg-loot-icons.zip
    python3 tools/build_foundry_module.py --set icons-256-v2 --out dist/rpg-loot-icons-v2.zip --id rpg-loot-icons-v2 --title "RPG Loot Icons V2"

Результат — zip, который можно распаковать в Data/modules/ или загрузить как модуль.
Также можно собрать без zip (папку): --no-zip --out dist/rpg-loot-icons
"""
import argparse
import json
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = {
    "id": "rpg-loot-icons",
    "title": "RPG Loot Icons — 4100 (DND5e)",
    "description": "4100 иконок предметов 256×256 WebP (quiet+vign) для DND5e. Чёрный фон, виньетка, единый стиль для грида лута. Источник: 82 листа RPG Loot Icons 01–41.",
    "version": "1.0.0",
    "compatibility": {"minimum": "11", "verified": "12"},
    "authors": [{"name": "RPG Loot Icons"}],
    "url": "https://github.com/aleksandarsolovjev35-spec/RPG-Loot-Icons-ALL",
    "manifest": "https://github.com/aleksandarsolovjev35-spec/RPG-Loot-Icons-ALL/releases/latest/download/module.json",
    "download": "https://github.com/aleksandarsolovjev35-spec/RPG-Loot-Icons-ALL/releases/latest/download/rpg-loot-icons.zip",
    "packs": [
        {
            "name": "loot-icons",
            "label": "Loot Icons (4100)",
            "path": "packs/loot-icons.db",
            "type": "Item",
            "system": "dnd5e"
        }
    ],
    "flags": {"rpg-loot-icons": {"sourceSet": "icons-256", "style": "quiet+vign"}}
}

def build(set_dir, out, module_id=None, title=None, no_zip=False, with_packs=False):
    set_path = os.path.join(ROOT, set_dir)
    man_path = os.path.join(set_path, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"no manifest at {man_path}")
    man = json.load(open(man_path))
    role = man.get("role", "")
    style = "+".join((man.get("style") or {}).get("presets", [])) or man.get("style", {}).get("title", "") or role
    size = man.get("size", 256)
    fmt = man.get("format", "webp")

    mod = dict(TEMPLATE)
    if module_id:
        mod["id"] = module_id
    if title:
        mod["title"] = title
    else:
        # auto title from style
        if style and "quiet" in style:
            mod["title"] = f"RPG Loot Icons — 4100 · {size}px · {style}"
    mod["version"] = man.get("generated", "1.0.0")[:10].replace("-", ".") if man.get("generated") else "1.0.0"
    # keep version semver-ish
    if not mod["version"][0].isdigit():
        mod["version"] = "1.0.0"
    # ensure version looks like x.y.z
    parts = mod["version"].split(".")
    while len(parts) < 3:
        parts.append("0")
    mod["version"] = ".".join(parts[:3])
    mod["flags"]["rpg-loot-icons"] = {
        "sourceSet": set_dir,
        "role": role,
        "style": style,
        "size": size,
        "format": fmt,
        "count": man.get("count", 0),
        "generated": man.get("generated", "")
    }

    # staging dir
    if no_zip:
        staging = out
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)
    else:
        staging = out + ".staging"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(os.path.join(staging, "icons"), exist_ok=True)

    root_for_copy = staging if no_zip else staging

    # copy icons
    print(f"copy {man.get('count', '?')} icons from {set_dir} -> {root_for_copy}/icons/ ...")
    # Use hardlink/copy via shutil.copytree with dirs_exist_ok
    src_icons = set_path
    dst_icons = os.path.join(root_for_copy, "icons")
    # copy only image subdirs + manifest, not the manifest root duplicated
    os.makedirs(dst_icons, exist_ok=True)
    for entry in man.get("icons", []):
        src = os.path.join(set_path, entry["file"])
        dst = os.path.join(dst_icons, entry["file"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    # also copy manifest for reference
    shutil.copy2(man_path, os.path.join(root_for_copy, "icons", "manifest.json"))
    # write module.json at root
    with open(os.path.join(root_for_copy, "module.json"), "w", encoding="utf-8") as f:
        json.dump(mod, f, indent=2, ensure_ascii=False)
        f.write("\n")
    # packs dir (empty placeholder or generate loot-icons.db if requested)
    packs_dir = os.path.join(root_for_copy, "packs")
    os.makedirs(packs_dir, exist_ok=True)
    if with_packs:
        # generate a minimal compendium db: one Item per icon, type loot, img = modules/<id>/icons/<file>
        db_path = os.path.join(packs_dir, "loot-icons.db")
        print(f"generate compendium {db_path} ({man.get('count')} Items) ...")
        with open(db_path, "w", encoding="utf-8") as out_db:
            for e in man.get("icons", []):
                name = f"{e['pack']} #{e['index']:03d}"
                # crude categorization from file path/pack number
                img = f"modules/{mod['id']}/icons/{e['file']}"
                doc = {
                    "name": name,
                    "type": "loot",
                    "img": img,
                    "system": {"description": {"value": f"<p>Icon {e['file']} — {size}×{size} {fmt}, {style}. Source {e.get('source','')}</p>"}},
                    "flags": {"rpg-loot-icons": {"pack": e.get("pack"), "part": e.get("part"), "index": e.get("index")}}
                }
                out_db.write(json.dumps(doc, ensure_ascii=False) + "\n")
    else:
        # placeholder so Foundry doesn't warn about missing pack file
        open(os.path.join(packs_dir, "loot-icons.db"), "a").close()

    # README
    with open(os.path.join(root_for_copy, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {mod['title']}\n\n")
        f.write(f"Source set: `{set_dir}` — {man.get('count')} icons, {size}×{size} {fmt}, style `{style}`, role `{role}`.\n\n")
        f.write("Install: unzip into `Data/modules/rpg-loot-icons/` (or your id) and enable the module. Icons are at `modules/rpg-loot-icons/icons/...`.\n")
        f.write("Use `modules/rpg-loot-icons/icons/<pack>/partN/icon_NNN.webp` as `Item.img` in DND5e.\n")
        f.write(f"Built from {man.get('generated','')}.\n")

    if no_zip:
        print(f"done: folder {out} ({man.get('count')} icons)")
        return out
    else:
        # zip
        if out.lower().endswith(".zip"):
            zip_path = out
        else:
            zip_path = out + ".zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"zip -> {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for dirpath, _, filenames in os.walk(root_for_copy):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, root_for_copy)
                    z.write(full, arc)
        shutil.rmtree(staging)
        print(f"done: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
        return zip_path

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_dir", default="icons-256", help="icon set dir (default icons-256)")
    ap.add_argument("--out", default="dist/rpg-loot-icons.zip", help="output zip or dir (default dist/rpg-loot-icons.zip)")
    ap.add_argument("--id", dest="module_id", default=None, help="module id (default rpg-loot-icons)")
    ap.add_argument("--title", default=None, help="module title")
    ap.add_argument("--no-zip", action="store_true", help="write a folder instead of a zip")
    ap.add_argument("--with-packs", action="store_true", help="generate packs/loot-icons.db with 4100 Items")
    args = ap.parse_args(argv)
    build(args.set_dir, args.out, args.module_id, args.title, args.no_zip, args.with_packs)

if __name__ == "__main__":
    main()
