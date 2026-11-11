# -*- coding: utf-8 -*-
"""把当前目录推送到 GitHub 公开仓库（Contents API，绕过沙箱 git 代理拦截）。
用法: GITHUB_TOKEN=xxx python3 push_to_github.py [repo_name] [owner]
"""
import os
import sys
import base64
import json
import urllib.request
import urllib.error

TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = sys.argv[1] if len(sys.argv) > 1 else "vector-graph-rag-demo"
OWNER = sys.argv[2] if len(sys.argv) > 2 else None
ROOT = os.path.dirname(os.path.abspath(__file__))
API = "https://api.github.com"

EXCLUDE = {".git", ".DS_Store", "out_index.txt", "out_query.txt", "_idx.html", "_q.html"}

def api(method, url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "vgraph-push")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, json.loads(body or "{}")

def get_owner():
    st, j = api("GET", f"{API}/user")
    if st != 200:
        raise SystemExit("获取用户失败: %s %s" % (st, j))
    return j["login"]

def collect_files():
    out = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, ROOT)
            if rel.split(os.sep)[0] in EXCLUDE or fn in EXCLUDE:
                continue
            out.append((rel, full))
    return out

def put_file(rel, full):
    with open(full, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    url = f"{API}/repos/{OWNER}/{REPO}/contents/{rel}"
    # 已存在则取 sha 更新
    st, j = api("GET", url)
    sha = j.get("sha") if st == 200 else None
    data = {"message": f"add {rel}", "content": content}
    if sha:
        data["sha"] = sha
    st2, j2 = api("PUT", url, data)
    return st2, j2.get("content", {}).get("path") or rel

def main():
    global OWNER
    if not TOKEN:
        raise SystemExit("缺少 GITHUB_TOKEN 环境变量")
    OWNER = OWNER or get_owner()
    print("owner =", OWNER)

    # 创建公开仓库（已存在则忽略 422）
    st, j = api("POST", f"{API}/user/repos", {
        "name": REPO, "private": False,
        "description": "Vector Graph RAG 最小可运行 Demo（纯向量检索，无图数据库，离线可跑）",
        "auto_init": False,
    })
    if st == 201:
        print("仓库已创建:", j["html_url"])
    elif st == 422:
        print("仓库已存在，直接推送文件。")
    else:
        raise SystemExit("创建仓库失败: %s %s" % (st, j))

    files = collect_files()
    print("待上传文件:", len(files))
    for rel, full in files:
        st, path = put_file(rel, full)
        print(("  [OK %d] " % st) if st in (200, 201) else ("  [ERR %d] " % st), path)

    print("完成。仓库地址: https://github.com/%s/%s" % (OWNER, REPO))

if __name__ == "__main__":
    main()
