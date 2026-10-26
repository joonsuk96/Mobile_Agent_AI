"""Read installed app labels from APK manifest/resources and cache locally."""
import difflib
import hashlib
import json
import re
import tempfile
import time
import unicodedata
from pathlib import Path

from .textio import read_local_text


def normalize(value):
    """Normalize spacing/case and decomposed Korean characters consistently."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in value if char.isalnum())


def load_alias_groups(path):
    """A reusable name dictionary, never a hardcoded app-to-package launch map."""
    if not path.is_file():
        return []
    try:
        data = json.loads(read_local_text(path))
    except (OSError, ValueError) as exc:
        raise ValueError("app_aliases.json을 읽지 못했어. JSON 형식을 확인해줘.") from exc
    if not isinstance(data, dict):
        raise ValueError("app_aliases.json은 앱 이름과 별칭 목록으로 작성해줘.")
    groups = []
    for canonical, aliases in data.items():
        if (not isinstance(canonical, str) or not canonical.strip()
                or not isinstance(aliases, list)
                or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
            raise ValueError("앱 별칭은 빈 값이 아닌 문자열 목록이어야 해.")
        group = list(dict.fromkeys([canonical, *aliases]))
        if any(not normalize(alias) for alias in group):
            raise ValueError("앱 별칭에는 글자나 숫자가 있어야 해.")
        groups.append(group)
    return groups


def expand_query(query, groups):
    """Expand exact alias equivalents in both directions; no extra AI calls."""
    terms = [query]
    keys = {normalize(query)}
    # Transitive aliases are allowed, e.g. two groups sharing an English name.
    changed = True
    while changed:
        changed = False
        for group in groups:
            if keys.intersection(normalize(alias) for alias in group):
                for alias in group:
                    key = normalize(alias)
                    if key not in keys:
                        terms.append(alias)
                        keys.add(key)
                        changed = True
    return terms

def read_apk_labels(path):
    from androguard.core.apk import APK
    from loguru import logger
    logger.disable("androguard")
    apk = APK(str(path))
    labels = []
    for locale in ("ko", None, "en"):
        try:
            label = apk.get_app_name(locale=locale)
        except Exception:
            continue
        if label and not label.startswith("@") and label not in labels:
            labels.append(label)
    return labels

class AppCatalog:
    def __init__(self, device, cache_dir, alias_path=None):
        self.device, self.cache_dir = device, Path(cache_dir)
        self.alias_path = Path(alias_path) if alias_path else self.cache_dir.parent / "app_aliases.json"
        self.apps, self.ready = {}, False
        self.confirmed_names_path = self.cache_dir / "confirmed-app-names.json"

    def build(self, refresh=False):
        if self.ready and not refresh:
            return
        components = self.device.launcher_components()
        fingerprint = self.device.text("shell", "getprop", "ro.build.fingerprint")
        user = self.device.text("shell", "am", "get-current-user")
        # Wireless ADB discovery addresses can change; use the hardware serial.
        identity = self.device.cache_identity()
        if not isinstance(identity, str) or not identity:
            identity = str(self.device.serial)
        key = hashlib.sha256(f"{identity}|{user}".encode()).hexdigest()[:20]
        legacy_key = hashlib.sha256(
            f"{self.device.serial}|{user}|{fingerprint}".encode()).hexdigest()[:20]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"apps-{key}.json"
        legacy_file = self.cache_dir / f"apps-{legacy_key}.json"
        self.confirmed_names_path = self.cache_dir / f"confirmed-app-names-{key}.json"
        legacy_confirmed = self.cache_dir / f"confirmed-app-names-{legacy_key}.json"
        if not self.confirmed_names_path.exists() and legacy_confirmed.is_file():
            # Preserve names the user previously confirmed for this same device.
            self.confirmed_names_path.write_text(
                read_local_text(legacy_confirmed), encoding="utf-8")
        cached = {}
        for candidate in (cache_file, legacy_file):
            try:
                data = json.loads(read_local_text(candidate))
                if isinstance(data, dict) and data.get("version") in (1, 2):
                    cached = data
                    break
            except (OSError, ValueError):
                continue
        entries = cached.get("apps", {})
        if not isinstance(entries, dict):
            entries = {}
        # Live components remove stale/uninstalled apps without reading every APK.
        apps = {}
        reused = 0
        extracted = 0

        def save_checkpoint():
            temporary_cache = cache_file.with_suffix(".tmp")
            temporary_cache.write_text(json.dumps({
                "version": 2, "device_id": identity, "android_user": user,
                "apps": {**entries, **apps},
            }, ensure_ascii=False), encoding="utf-8")
            # Windows scanners/sync clients can briefly hold the destination.
            # Preserve atomic replacement; permanent permission errors still fail.
            for attempt in range(3):
                try:
                    temporary_cache.replace(cache_file)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.05 * (attempt + 1))

        for index, (package, component) in enumerate(sorted(components.items()), 1):
            old = entries.get(package, {})
            valid = (
                isinstance(old, dict)
                and isinstance(old.get("labels"), list)
                and all(isinstance(label, str) for label in old["labels"])
                and isinstance(old.get("name"), str)
            )
            if not refresh and valid:
                # Negative caching matters: unresolved/failed labels are reusable too.
                apps[package] = {**old, "package": package, "component": component}
                reused += 1
                continue
            print(f"앱 이름 읽는 중 [{index}/{len(components)}]: {package}")
            paths, labels, error_type = [], [], ""
            try:
                paths = self.device.apk_paths(package)
                base = next((path for path in paths if path.endswith("/base.apk")),
                            paths[0] if paths else None)
                if base:
                    with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
                        apk_file = Path(temporary) / "base.apk"
                        self.device.pull(base, apk_file)
                        labels = read_apk_labels(apk_file)
            except Exception as exc:
                error_type = type(exc).__name__
                # Keep paths and failure status: do not retry next run automatically.
                print(f"앱 이름 생략: {package} ({error_type}); 패키지 정보로 검색할게.")
            apps[package] = {
                "package": package, "component": component,
                "name": labels[0] if labels else package,
                "labels": labels, "paths": paths,
                "label_status": "resolved" if labels else "error" if error_type else "unresolved",
                "label_error": error_type,
            }
            extracted += 1
            save_checkpoint()
        self.apps, self.ready = apps, True
        # Final cache excludes uninstalled apps, while mid-index checkpoints allow resume.
        entries = {}
        save_checkpoint()
        print(f"앱 목록 준비: 캐시 {reused}개 재사용, 새로 확인 {extracted}개.")

    def search(self, query="", offset=0, limit=20, refresh=False, catalog_offset=0):
        self.build(refresh=refresh)
        # Read on each search: dictionary changes apply to existing app caches.
        groups = load_alias_groups(self.alias_path)
        expanded_queries = expand_query(query, groups)
        confirmed = self._load_confirmed_names().get(normalize(query))
        ranked = []
        for app in self.apps.values():
            full_names = {normalize(name) for name in [app["name"], *app["labels"], app["package"]]}
            if not app["labels"]:
                # APK labels may be unavailable; retain the app's terminal ID.
                full_names.add(normalize(app["package"].rsplit(".", 1)[-1]))
            names = list(dict.fromkeys([
                app["name"], *app["labels"], app["package"],
                *app["package"].split("."),
            ]))
            score, matched_name, matched_query, match_type = 0.0, "", "", ""
            if normalize(query):
                for term in expanded_queries:
                    term_key = normalize(term)
                    for name in names:
                        candidate = normalize(name)
                        if not candidate:
                            continue
                        if term_key == candidate:
                            # A package segment is weaker than the actual app label.
                            # YouTube Music's package contains "youtube" too.
                            match, kind = ((1.0, "exact") if candidate in full_names
                                           else (0.88, "package_segment"))
                        elif term_key in candidate:
                            match, kind = 0.9, "partial"
                        else:
                            match = difflib.SequenceMatcher(None, term_key, candidate).ratio()
                            kind = "fuzzy"
                        if normalize(term) != normalize(query):
                            kind = "alias_" + kind
                        if match > score:
                            score, matched_name, matched_query, match_type = match, name, term, kind
                if confirmed and app["package"] == confirmed["package"]:
                    score, matched_name, matched_query, match_type = (
                        1.1, app["name"], query, "user_confirmed")
                if score < 0.55:
                    continue
            else:
                match_type = "listing"
            aliases = list(dict.fromkeys([
                alias for group in groups
                if any(normalize(name) == normalize(alias)
                       for name in names for alias in group)
                for alias in group
            ]))
            entry = {key: app[key] for key in ("name", "labels", "package")}
            entry.update({
                "aliases": aliases, "matched_name": matched_name,
                "matched_query": matched_query, "match_type": match_type,
            })
            ranked.append((score, entry))
        ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
        items = [entry for _, entry in ranked]
        offset, limit = max(0, offset), max(1, min(30, limit))
        strong = any(entry["match_type"] in {
            "exact", "alias_exact", "user_confirmed"
        } for entry in items)
        needs_semantic = bool(normalize(query)) and not strong
        semantic_page = (self.semantic_catalog_page(catalog_offset)
                         if needs_semantic else None)
        return {
            "resolution_mode": "semantic_required" if needs_semantic else "local",
            "semantic_catalog": semantic_page,
            "semantic_instruction": (
                "문자열 검색만으로 앱을 확정할 수 없다. 아래 실제 설치 앱의 영문 이름, "
                "labels, package를 한국어 요청의 번역명/발음/브랜드와 의미로 비교하라. "
                "확실히 동일한 앱이면 반환된 package로 실행할 수 있다. "
                "비슷한 용도라는 이유만으로 다른 앱을 대신 선택하지 마라. "
                "모호하면 confirm_app_choice로 물어보고, 후보가 없으면 "
                "semantic_catalog.next_offset을 catalog_offset에 넣어 다음 페이지를 확인하라."
            ) if needs_semantic else "",
            "query": query, "expanded_queries": expanded_queries,
            "total_matches": len(items), "apps": items[offset:offset+limit],
            "next_offset": offset+limit if offset+limit < len(items) else None,
            "unresolved_name_count": sum(not app["labels"] for app in self.apps.values()),
            "note": "별칭은 검색 보조 정보다. 여러 후보가 나오면 사용자에게 확인한다.",
        }


    def _load_confirmed_names(self):
        try:
            saved = json.loads(read_local_text(self.confirmed_names_path))
        except (OSError, ValueError):
            return {}
        if not isinstance(saved, dict) or saved.get("version") != 1:
            return {}
        matches = saved.get("matches", {})
        if not isinstance(matches, dict):
            return {}
        # Only current, installed packages and explicit user confirmations survive.
        return {
            key: entry for key, entry in matches.items()
            if isinstance(key, str) and isinstance(entry, dict)
            and entry.get("confirmed_by") == "user"
            and isinstance(entry.get("package"), str)
            and entry["package"] in self.apps
        }

    def remember_confirmed_name(self, query, package):
        """Called only after a real terminal confirmation, never by model assertion."""
        self.get(package)
        key = normalize(query)
        if not key:
            raise ValueError("기억할 앱 이름이 비어 있어.")
        matches = self._load_confirmed_names()
        matches[key] = {"query": query, "package": package, "confirmed_by": "user"}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.confirmed_names_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(
            {"version": 1, "matches": matches}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.confirmed_names_path)

    def semantic_catalog_page(self, offset=0, limit=50):
        """Compact real app metadata for the existing agent's cross-language reasoning."""
        ordered = sorted(self.apps.values(), key=lambda app: (app["name"], app["package"]))
        offset = max(0, offset)
        limit = max(1, min(50, limit))
        return {
            "apps": [{key: app[key] for key in ("name", "labels", "package")}
                     for app in ordered[offset:offset + limit]],
            "total_installed": len(ordered),
            "next_offset": offset + limit if offset + limit < len(ordered) else None,
        }

    def get(self, package):
        if not self.ready or package not in self.apps:
            raise ValueError("search_apps가 반환한 실제 package를 사용해줘.")
        return self.apps[package]
