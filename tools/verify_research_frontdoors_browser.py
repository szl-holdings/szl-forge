#!/usr/bin/env python3
"""Render two exact native front doors with local, read-only fixtures.

No application, trained checkpoint, scoring endpoint or publication is started.
The hidden table/receipt stress state is explicitly a local layout fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
WIDTHS = (320, 390, 768, 1280, 1920)
UNAVAILABLE = {
    "ready": False, "state": "UNAVAILABLE", "status": "UNAVAILABLE",
    "error": "Local read-only layout fixture; no model runtime was started.",
}


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "LOCAL_PRESENTATION_ONLY", "model_inference_performed": False,
        "operational_writes_performed": False, "deployed_runtime_verified": False,
        "cases": [],
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=os.environ.get("CHROME_PATH"),
            args=["--disable-dev-shm-usage", "--use-gl=angle", "--use-angle=swiftshader",
                  "--enable-unsafe-swiftshader"],
        )
        for name, source in (
            ("oac-health", REPO / "spaces/oac-system-health-lab"),
            ("foundation", REPO / "spaces/szl-foundation-confirmation/web"),
        ):
            requests, writes, errors, outside = [], [], [], []
            files = {"/": ((source / "index.html").read_bytes(), "text/html; charset=utf-8")}
            icon = re.search(rb'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600".*?</svg>', files["/"][0], re.DOTALL)
            icon_exact = icon is not None and hashlib.sha256(icon.group(0) + b"\n").hexdigest() == "6e2b7d72a3af3d9602b79de41df0194ede6496a9661f057108ec224c0efc92e6"
            if name == "foundation":
                files["/style.css"] = ((source / "style.css").read_bytes(), "text/css; charset=utf-8")
                files["/app.js"] = ((source / "app.js").read_bytes(), "text/javascript; charset=utf-8")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    path = urlsplit(self.path).path
                    requests.append(path)
                    if path in files:
                        body, mime = files[path]
                        code = 200
                    else:
                        body, mime, code = json.dumps(UNAVAILABLE).encode(), "application/json", 503
                    self.send_response(code)
                    # MIME values come only from the closed literal route table above.
                    self.send_header("Content-Type", mime)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)

                def do_POST(self):
                    writes.append(self.path)
                    self.send_error(405)

                do_PUT = do_POST
                do_PATCH = do_POST
                do_DELETE = do_POST

                def log_message(self, *_):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            context = browser.new_context(reduced_motion="reduce")

            def route(request):
                if request.request.method != "GET":
                    writes.append(request.request.url)
                    request.abort()
                elif request.request.url.startswith(origin + "/"):
                    request.continue_()
                else:
                    outside.append(request.request.url)
                    request.fulfill(status=503, content_type="application/json", body=json.dumps(UNAVAILABLE))

            context.route("**/*", route)
            page = context.new_page()
            page.set_default_timeout(20000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            rows = []
            for width in WIDTHS:
                page.set_viewport_size({"width": width, "height": 1000})
                page.goto(origin + "/", wait_until="domcontentloaded")
                if name == "foundation":
                    page.wait_for_function("document.querySelector('#connection').textContent === 'Workbench unavailable'")
                else:
                    page.wait_for_function("document.querySelector('#service-status').dataset.state === 'unavailable'")
                page.keyboard.press("Tab")
                first = page.evaluate("document.activeElement.matches('a.skip-link')")
                page.keyboard.press("Enter")
                target_focus = page.evaluate("""() => {
                    const target = document.querySelector(document.querySelector('.skip-link').hash);
                    return target === document.activeElement;
                }""")
                layout = page.evaluate("""() => {
                    const shown = e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden';
                    const targets = [...document.querySelectorAll('button:not(:disabled),select,input:not([type=checkbox]),summary,footer a,header a')].filter(shown);
                    const small = targets.filter(e => e.getBoundingClientRect().height < 43.5)
                      .map(e => ({id:e.id,tag:e.tagName,height:e.getBoundingClientRect().height}));
                    return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,small,
                      h1:document.querySelector('h1').textContent.trim(),
                      overflow:[...document.querySelectorAll('body *')].filter(e=>shown(e) && e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,id:e.id,className:e.className,right:e.getBoundingClientRect().right})).slice(0,8),
                      panel:getComputedStyle(document.querySelector('.card,.panel')).backgroundImage,
                      iconCount:document.querySelectorAll('header svg').length};
                }""")
                unavailable = page.locator("#run-trial" if name == "foundation" else "#score-button").is_disabled()
                rows.append({"width": width, "skip_first": first, "skip_focus": target_focus,
                             "layout": layout, "unavailable_action_disabled": unavailable})
                if width in (320, 1280):
                    page.evaluate("window.scrollTo(0,0)")
                    page.screenshot(path=str(output / f"{name}-{width}.png"), full_page=False)

            stress = None
            if name == "foundation":
                page.set_viewport_size({"width": 320, "height": 1000})
                # State produced solely to inspect an overflow region. No trial request.
                page.evaluate("""() => {
                    document.querySelector('#history-section').hidden = false;
                    document.querySelector('#receipt-details').hidden = false;
                    document.querySelector('#receipt-json').textContent = JSON.stringify({
                      fixture:'LOCAL_LAYOUT_ONLY_NO_INFERENCE',sha:'a'.repeat(64),lines:Array(40).fill('Layout fixture')},null,2);
                    const row = document.createElement('tr');
                    for(let i=0;i<7;i++){const td=document.createElement('td');td.textContent='LOCAL LAYOUT FIXTURE '+i;row.append(td);}
                    document.querySelector('#history').append(row);
                }""")
                region = page.locator(".table-scroll")
                region.focus()
                before = region.evaluate("e=>e.scrollLeft")
                page.keyboard.press("ArrowRight")
                page.wait_for_timeout(150)
                table = region.evaluate("e=>({left:e.scrollLeft,client:e.clientWidth,width:e.scrollWidth,outline:getComputedStyle(e).outlineWidth,role:e.getAttribute('role')})")
                summary = page.locator("#receipt-details summary")
                summary.focus()
                page.keyboard.press("Space")
                opened = page.locator("#receipt-details").evaluate("e=>e.open")
                receipt = page.locator("#receipt-json")
                receipt.focus()
                page.keyboard.press("PageDown")
                page.wait_for_function("document.querySelector('#receipt-json').scrollTop > 0")
                receipt_scroll = receipt.evaluate("e=>({focused:e===document.activeElement,top:e.scrollTop,outline:getComputedStyle(e).outlineWidth})")
                stress = {"scope": "LOCAL_LAYOUT_ONLY_NO_INFERENCE", "table": table,
                          "table_keyboard_scrolled": table["left"] > before,
                          "receipt_opened_with_space": opened, "receipt": receipt_scroll,
                          "no_page_overflow": page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")}
                # A previous Ready label must clear when the next read fails.
                page.evaluate("""() => {
                    document.querySelector('#sources').textContent='Ready';
                    document.querySelector('#checkpoints').textContent='3';
                    document.querySelector('#loaded').textContent='3 / 3';
                }""")
                page.route("**/api/status", lambda route: route.abort())
                page.locator("#refresh-status").click()
                page.wait_for_function("document.querySelector('#sources').textContent === 'Unavailable'")
                stress["stale_ready_cleared"] = page.locator("#checkpoints").inner_text() == "—" and page.locator("#loaded").inner_text() == "—"
                stress["bounded_get_abort"] = page.evaluate("""async () => {
                    const savedFetch=window.fetch, savedTimer=window.setTimeout;
                    let aborted=false, requestedDelay=null;
                    window.setTimeout=(fn,delay,...args)=>{requestedDelay=delay;return savedTimer(fn,delay===10000?1:delay,...args)};
                    window.fetch=(_url,options)=>new Promise((_resolve,reject)=>{
                      options.signal.addEventListener('abort',()=>{aborted=true;reject(new DOMException('Local timeout fixture','AbortError'))},{once:true});
                    });
                    try { await refresh(); return {aborted,deadline:requestedDelay,unavailable:document.querySelector('#sources').textContent==='Unavailable'}; }
                    finally { window.fetch=savedFetch; window.setTimeout=savedTimer; }
                }""")

            page.emulate_media(forced_colors="active", reduced_motion="reduce")
            page.keyboard.press("Tab")
            page.locator(".skip-link").focus()
            forced = page.evaluate("""() => ({
                outline:getComputedStyle(document.activeElement).outlineStyle,
                width:getComputedStyle(document.activeElement).outlineWidth,
                panel:getComputedStyle(document.querySelector('.card,.panel')).backgroundColor,
                image:getComputedStyle(document.querySelector('.card,.panel')).backgroundImage
            })""")
            passed = all(r["skip_first"] and r["skip_focus"] and r["unavailable_action_disabled"]
                         and r["layout"]["scrollWidth"] <= r["width"] + 1
                         and not r["layout"]["small"] and r["layout"]["iconCount"] == 1 for r in rows)
            if stress:
                passed = passed and stress["table_keyboard_scrolled"] and stress["receipt_opened_with_space"] and stress["receipt"]["focused"] and stress["no_page_overflow"] and stress["stale_ready_cleared"] and stress["bounded_get_abort"]["aborted"] and stress["bounded_get_abort"]["deadline"] == 10000 and stress["bounded_get_abort"]["unavailable"]
            passed = passed and icon_exact and not errors and not writes and not outside and forced["outline"] == "solid" and forced["width"] == "3px" and forced["image"] == "none"
            record = {"name": name, "passed": passed, "widths": rows, "stress": stress,
                      "forced_colors": forced, "accepted_icon_exact": icon_exact, "page_errors": errors, "attempted_writes": writes,
                      "blocked_external_gets": outside, "served_paths": sorted(set(requests)),
                      "source_hashes": {path: hashlib.sha256(body).hexdigest() for path, (body, _) in files.items()}}
            report["cases"].append(record)
            print(json.dumps({"name": name, "passed": passed, "page_errors": errors,
                              "small_targets": [r for r in rows if r["layout"]["small"]]}), flush=True)
            context.close()
            server.shutdown()
            server.server_close()
            thread.join()
        browser.close()
    report["passed"] = all(case["passed"] for case in report["cases"])
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().output))
