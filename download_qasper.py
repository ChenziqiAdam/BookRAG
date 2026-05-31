import json, os, time, requests

data = json.load(open("datasets/Qasper.json"))
pdf_dir = "qasper"
os.makedirs(pdf_dir, exist_ok=True)

ids = list({entry["doc_path"].split("/")[-1].replace(".pdf","") for entry in data})

for arxiv_id in ids:
    out = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
    if os.path.exists(out):
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = requests.get(url)
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {arxiv_id}")
    time.sleep(3)  # be polite to arXiv