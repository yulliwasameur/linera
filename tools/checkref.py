import json, sys, urllib.request, urllib.parse, re, time
def get(u):
    req=urllib.request.Request(u, headers={"User-Agent":"linera-refcheck/1.0 (mailto:noreply@example.org)"})
    return urllib.request.urlopen(req, timeout=30).read().decode()
def doi(d):
    m=json.loads(get("https://api.crossref.org/works/"+urllib.parse.quote(d)))["message"]
    a=", ".join(f"{x.get('given','')} {x.get('family','')}" for x in m.get("author",[])[:8])
    return f"DOI {d}: {m.get('title',[''])[0]} | {a} | {m.get('container-title',[''])[0]} | {m.get('issued',{}).get('date-parts',[[None]])[0]}"
def title(t):
    m=json.loads(get("https://api.crossref.org/works?rows=3&query.bibliographic="+urllib.parse.quote(t)))["message"]["items"]
    return "\n".join(f"  ? {x.get('title',[''])[0]} | {x.get('DOI')} | {x.get('container-title',[''])[:1]} | {x.get('issued',{}).get('date-parts')}" for x in m)
def arxiv(i):
    x=get("https://export.arxiv.org/api/query?id_list="+i)
    ti=re.findall(r"<title>(.*?)</title>", x, re.S); au=re.findall(r"<name>(.*?)</name>", x)
    return f"arXiv {i}: {ti[1].strip() if len(ti)>1 else 'NOT FOUND'} | {', '.join(au[:6])}"
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    kind,val=line.split(" ",1)
    try:
        print({"doi":doi,"title":title,"arxiv":arxiv}[kind](val))
    except Exception as e: print("ERR", line, e)
    time.sleep(0.5)
