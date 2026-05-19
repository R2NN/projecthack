from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


STATUSES = [
    "ok",
    "fix",
    "reject_blur",
    "reject_cut",
    "reject_not_name",
    "reject_multi_line",
    "reject_unclear",
]

APPROVED_STATUSES = {"ok", "fix"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    value = row.get("priority_rank", "").strip()
    if value:
        return value
    return "|".join(
        [
            row.get("rel_path", "").strip(),
            row.get("source_id", "").strip(),
            row.get("row_index", "").strip(),
            row.get("line_index", "").strip(),
        ]
    )


def normalize_status(value: str) -> str:
    value = str(value or "").strip()
    return value if value in STATUSES else ""


class ReviewStore:
    def __init__(self, dataset_dir: Path, queue_csv: Path, output_csv: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.queue_csv = queue_csv.resolve()
        self.output_csv = output_csv.resolve()
        self.lock = threading.Lock()

        base_rows = read_rows(self.queue_csv)
        existing_rows = read_rows(self.output_csv) if self.output_csv.exists() else []
        existing = {row_key(row): row for row in existing_rows}
        fieldnames = list(base_rows[0].keys()) if base_rows else []
        for column in ("status", "review_note"):
            if column not in fieldnames:
                fieldnames.append(column)
        self.fieldnames = fieldnames

        self.rows: list[dict[str, str]] = []
        for base in base_rows:
            merged = {name: base.get(name, "") for name in self.fieldnames}
            current = existing.get(row_key(base), {})
            for name in self.fieldnames:
                if name in current and current[name] != "":
                    merged[name] = current[name]
            if not merged.get("reviewed_label"):
                merged["reviewed_label"] = merged.get("proposed_label", "")
            if not merged.get("status"):
                approved = str(merged.get("approved", "")).strip().lower()
                if approved in {"1", "true", "yes", "y", "ok", "approved"}:
                    merged["status"] = "ok"
            self.rows.append(merged)
        self.key_to_index = {row_key(row): index for index, row in enumerate(self.rows)}

    def save(self) -> None:
        write_rows(self.output_csv, self.rows, self.fieldnames)

    def progress(self) -> dict[str, Any]:
        reviewed = [row for row in self.rows if normalize_status(row.get("status", ""))]
        approved = [row for row in self.rows if row.get("approved", "").strip() == "1"]
        rejected = [row for row in reviewed if row.get("approved", "").strip() != "1"]
        return {
            "total": len(self.rows),
            "reviewed": len(reviewed),
            "approved": len(approved),
            "rejected": len(rejected),
            "remaining": len(self.rows) - len(reviewed),
            "output_csv": str(self.output_csv),
        }

    def item(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "index": index,
            "key": row_key(row),
            "priority_rank": row.get("priority_rank", ""),
            "line_url": image_url(row.get("rel_path", "")),
            "field_url": image_url(row.get("field_rel_path", "")),
            "tag_url": image_url(row.get("tag_rel_path", "")),
            "reviewed_label": row.get("reviewed_label", ""),
            "proposed_label": row.get("proposed_label", ""),
            "full_product_name": row.get("full_product_name", ""),
            "status": row.get("status", ""),
            "approved": row.get("approved", ""),
            "review_note": row.get("review_note", ""),
            "source_id": row.get("source_id", ""),
            "line_index": row.get("line_index", ""),
            "line_count": row.get("line_count", ""),
            "quality_score": row.get("quality_score", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
        }

    def list_items(self, offset: int, limit: int, mode: str) -> dict[str, Any]:
        if mode == "unreviewed":
            indexes = [idx for idx, row in enumerate(self.rows) if not normalize_status(row.get("status", ""))]
        elif mode == "approved":
            indexes = [idx for idx, row in enumerate(self.rows) if row.get("approved", "").strip() == "1"]
        elif mode == "rejected":
            indexes = [
                idx
                for idx, row in enumerate(self.rows)
                if normalize_status(row.get("status", "")) and row.get("approved", "").strip() != "1"
            ]
        else:
            indexes = list(range(len(self.rows)))
        selected = indexes[offset : offset + limit]
        return {
            "items": [self.item(index) for index in selected],
            "offset": offset,
            "limit": limit,
            "mode": mode,
            "filtered_total": len(indexes),
            "progress": self.progress(),
        }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = str(payload.get("key", "")).strip()
        if key not in self.key_to_index:
            raise KeyError(f"Unknown row key: {key}")
        status = normalize_status(str(payload.get("status", "")))
        if not status:
            raise ValueError(f"Status must be one of: {', '.join(STATUSES)}")
        label = str(payload.get("reviewed_label", "")).replace("\r", " ").replace("\n", " ").strip()
        if status in APPROVED_STATUSES and not label:
            raise ValueError("Approved rows must have reviewed_label")

        with self.lock:
            row = self.rows[self.key_to_index[key]]
            row["status"] = status
            row["reviewed_label"] = label
            row["approved"] = "1" if status in APPROVED_STATUSES else "0"
            row["review_note"] = str(payload.get("review_note", "")).replace("\r", " ").replace("\n", " ").strip()
            self.save()
            return {"item": self.item(self.key_to_index[key]), "progress": self.progress()}

    def resolve_image(self, rel_path: str) -> Path:
        rel_path = unquote(rel_path).replace("\\", "/").lstrip("/")
        path = (self.dataset_dir / rel_path).resolve()
        if self.dataset_dir not in path.parents and path != self.dataset_dir:
            raise ValueError("Path escapes dataset directory")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        return path


def image_url(rel_path: str) -> str:
    return f"/image?path={quote(str(rel_path or '').replace('\\', '/'))}"


def build_html() -> str:
    statuses_json = json.dumps(STATUSES, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Line OCR Review</title>
  <style>
    :root {{ color-scheme: light; --bg: #f4f6f8; --panel: #ffffff; --line: #d7dee8; --text: #18222f; --muted: #667487; --accent: #1f6feb; --bad: #b42318; --ok: #087443; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: #ffffff; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: 18px; }}
    .counter {{ color: var(--muted); font-size: 13px; }}
    select, input, textarea, button {{ font: inherit; }}
    button {{ border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 8px 10px; cursor: pointer; }}
    button:hover {{ border-color: #9aa8ba; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    button.ok {{ background: #e9f7ef; border-color: #9bd7b5; color: var(--ok); }}
    button.reject {{ background: #fff1f0; border-color: #ffc9c5; color: var(--bad); }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 16px; }}
    .layout {{ display: grid; grid-template-columns: minmax(360px, 1.25fr) minmax(340px, 0.75fr); gap: 16px; align-items: start; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .line-crop {{ width: 100%; min-height: 150px; display: flex; align-items: center; justify-content: center; background: #fff; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
    .line-crop img {{ width: 100%; image-rendering: auto; }}
    .context-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
    .context img {{ width: 100%; max-height: 280px; object-fit: contain; background: #fff; border: 1px solid var(--line); border-radius: 6px; }}
    .label {{ color: var(--muted); font-size: 12px; margin: 0 0 6px; }}
    textarea {{ width: 100%; min-height: 92px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 10px; font-size: 18px; line-height: 1.35; }}
    .hint {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; font-size: 13px; }}
    .meta div {{ padding: 8px; background: #f7f9fb; border: 1px solid #e4e9ef; border-radius: 6px; }}
    .actions {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .wide {{ grid-column: 1 / -1; }}
    .full-name {{ padding: 10px; background: #f7f9fb; border: 1px solid #e4e9ef; border-radius: 6px; line-height: 1.35; }}
    .status-line {{ margin-top: 10px; color: var(--muted); font-size: 13px; min-height: 18px; }}
    @media (max-width: 900px) {{ .layout, .context-grid {{ grid-template-columns: 1fr; }} header {{ flex-wrap: wrap; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Line OCR Review</h1>
    <button id="prev">Назад</button>
    <button id="next">Дальше</button>
    <select id="mode">
      <option value="unreviewed">Непроверенные</option>
      <option value="all">Все</option>
      <option value="approved">Approved</option>
      <option value="rejected">Rejected</option>
    </select>
    <span class="counter" id="counter"></span>
  </header>
  <main>
    <div class="layout">
      <section class="panel">
        <p class="label">Строковый кроп</p>
        <div class="line-crop"><img id="lineImg" alt=""></div>
        <div class="context-grid">
          <div class="context">
            <p class="label">Весь product_name crop</p>
            <img id="fieldImg" alt="">
          </div>
          <div class="context">
            <p class="label">Ценник целиком</p>
            <img id="tagImg" alt="">
          </div>
        </div>
      </section>
      <section class="panel">
        <p class="label">Текст для обучения</p>
        <textarea id="labelText" spellcheck="false"></textarea>
        <p class="label">Полное название-подсказка</p>
        <div class="full-name" id="fullName"></div>
        <div class="meta">
          <div>rank: <b id="rank"></b></div>
          <div>source: <b id="source"></b></div>
          <div>line: <b id="lineNo"></b></div>
          <div>quality: <b id="quality"></b></div>
        </div>
        <div class="actions">
          <button class="ok primary" id="saveOk">OK / Enter</button>
          <button class="ok" id="saveFix">Fix</button>
          <button class="reject" data-status="reject_blur">Blur</button>
          <button class="reject" data-status="reject_cut">Cut</button>
          <button class="reject" data-status="reject_not_name">Not name</button>
          <button class="reject" data-status="reject_multi_line">Multi-line</button>
          <button class="reject wide" data-status="reject_unclear">Unclear</button>
        </div>
        <p class="hint">Правило: в поле пишем ровно то, что видно на строковом кропе. Не восстанавливаем товар по смыслу. Если строка плохая, жмем reject.</p>
        <div class="status-line" id="statusLine"></div>
      </section>
    </div>
  </main>
  <script>
    const STATUSES = {statuses_json};
    let items = [];
    let cursor = 0;
    let offset = 0;
    const limit = 50;

    const el = id => document.getElementById(id);
    const setStatus = text => el('statusLine').textContent = text;

    async function api(path, options = {{}}) {{
      const response = await fetch(path, options);
      if (!response.ok) {{
        const text = await response.text();
        throw new Error(text || response.statusText);
      }}
      return response.json();
    }}

    async function loadBatch(newOffset = 0) {{
      offset = Math.max(0, newOffset);
      const mode = el('mode').value;
      const data = await api(`/api/items?offset=${{offset}}&limit=${{limit}}&mode=${{encodeURIComponent(mode)}}`);
      items = data.items;
      cursor = 0;
      render();
      renderProgress(data.progress, data.filtered_total);
    }}

    function renderProgress(progress, filteredTotal) {{
      el('counter').textContent = `проверено ${{progress.reviewed}}/${{progress.total}}, approved ${{progress.approved}}, осталось ${{progress.remaining}}, в фильтре ${{filteredTotal}}`;
    }}

    function current() {{
      return items[cursor] || null;
    }}

    function render() {{
      const item = current();
      if (!item) {{
        el('lineImg').removeAttribute('src');
        el('fieldImg').removeAttribute('src');
        el('tagImg').removeAttribute('src');
        el('labelText').value = '';
        el('fullName').textContent = 'Нет строк в текущем фильтре.';
        setStatus('');
        return;
      }}
      el('lineImg').src = item.line_url;
      el('fieldImg').src = item.field_url;
      el('tagImg').src = item.tag_url;
      el('labelText').value = item.reviewed_label || item.proposed_label || '';
      el('fullName').textContent = item.full_product_name || '';
      el('rank').textContent = item.priority_rank || '';
      el('source').textContent = item.source_id || '';
      el('lineNo').textContent = `${{item.line_index || '?'}}/${{item.line_count || '?'}}`;
      el('quality').textContent = item.quality_score || '';
      setStatus(item.status ? `уже размечено: ${{item.status}}` : '');
      el('labelText').focus();
      el('labelText').select();
    }}

    async function save(status) {{
      const item = current();
      if (!item) return;
      const payload = {{
        key: item.key,
        status,
        reviewed_label: el('labelText').value.trim(),
        review_note: ''
      }};
      const data = await api('/api/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      renderProgress(data.progress, data.progress.total);
      setStatus(`сохранено: ${{status}}`);
      next();
    }}

    function next() {{
      if (cursor + 1 < items.length) {{
        cursor += 1;
        render();
      }} else {{
        loadBatch(offset + limit).catch(error => setStatus(error.message));
      }}
    }}

    function prev() {{
      if (cursor > 0) {{
        cursor -= 1;
        render();
      }} else if (offset > 0) {{
        loadBatch(offset - limit).catch(error => setStatus(error.message));
      }}
    }}

    el('saveOk').addEventListener('click', () => save('ok').catch(error => setStatus(error.message)));
    el('saveFix').addEventListener('click', () => save('fix').catch(error => setStatus(error.message)));
    document.querySelectorAll('[data-status]').forEach(button => {{
      button.addEventListener('click', () => save(button.dataset.status).catch(error => setStatus(error.message)));
    }});
    el('next').addEventListener('click', next);
    el('prev').addEventListener('click', prev);
    el('mode').addEventListener('change', () => loadBatch(0).catch(error => setStatus(error.message)));
    document.addEventListener('keydown', event => {{
      if (event.key === 'ArrowRight') next();
      if (event.key === 'ArrowLeft') prev();
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) save('ok').catch(error => setStatus(error.message));
      if (event.altKey && event.key.toLowerCase() === 'b') save('reject_blur').catch(error => setStatus(error.message));
      if (event.altKey && event.key.toLowerCase() === 'c') save('reject_cut').catch(error => setStatus(error.message));
      if (event.altKey && event.key.toLowerCase() === 'n') save('reject_not_name').catch(error => setStatus(error.message));
      if (event.altKey && event.key.toLowerCase() === 'm') save('reject_multi_line').catch(error => setStatus(error.message));
    }});
    loadBatch(0).catch(error => setStatus(error.message));
  </script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(build_html(), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/items":
            query = parse_qs(parsed.query)
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["50"])[0])
            mode = query.get("mode", ["unreviewed"])[0]
            self.send_json(self.store.list_items(offset, min(max(limit, 1), 200), mode))
            return
        if parsed.path == "/api/progress":
            self.send_json(self.store.progress())
            return
        if parsed.path == "/image":
            query = parse_qs(parsed.query)
            try:
                path = self.store.resolve_image(query.get("path", [""])[0])
                data = path.read_bytes()
            except Exception as error:
                self.send_text(str(error), status=404)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_text("Not found", status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self.send_text("Not found", status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(self.store.update(payload))
        except Exception as error:
            self.send_text(str(error), status=400)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local UI for reviewing OCR line crops.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("repro_outputs/ppocr_finetune/real_product_name_v1"),
    )
    parser.add_argument(
        "--queue-csv",
        type=Path,
        default=Path("repro_outputs/ppocr_finetune/real_product_name_v1/line_candidates/review_queue_priority.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("repro_outputs/ppocr_finetune/real_product_name_v1/line_candidates/review_queue_priority_reviewed.csv"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ReviewStore(args.dataset_dir, args.queue_csv, args.output_csv)
    ReviewHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"review_url=http://{args.host}:{args.port}/")
    print(f"output_csv={store.output_csv}")
    print(json.dumps(store.progress(), ensure_ascii=False, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
