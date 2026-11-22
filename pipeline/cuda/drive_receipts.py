"""Run the unmodified drive command, streaming each durable stage and immutable release receipt."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path(os.environ['RS_INSTALL']);dst=root/'bench/results';base=root/'work/drives/hw4-cuda-224-254';manifest=base/'manifest.json'; seenfile=dst/'emitted.json';seen=set(json.loads(seenfile.read_text())) if seenfile.is_file() else set()
def emit(kind,**data):print(json.dumps(dict(receipt=kind,**data)),flush=True)
def snapshot():
    if not manifest.exists():return
    m=json.loads(manifest.read_text());
    for k,r in {**m.get('stages',{}),**m.get('units',{})}.items():
        key=k+str(r.get('finish'))
        if r.get('status')=='failed' and key not in seen:
            log=Path(r['log']);print(log.read_text(errors='replace')[-6500:],file=sys.stderr,flush=True);emit('stage_error',stage=k,detail=r.get('detail'))
        elif r.get('status')=='done' and key not in seen:
            emit('stage_done',stage=k,seconds=r['duration_s'],finish=r['finish'])
            if k=='publish' or k.endswith('-publish'):
                gen=Path(r['generation']);file=gen/('work/run/published.json' if k=='publish' else 'publication.json');pub=json.loads(file.read_text());emit('drive_release',stage=k,directory=pub['directory'],sha=pub['content_sha256'])
        else:continue
        seen.add(key);seenfile.write_text(json.dumps(sorted(seen)))
cmd=[sys.executable,'pipeline/run/drive.py','run','clips','--name','hw4-cuda-224-254','--window','224','254','--quality','full','--clip','all']
with (dst/'drive.log').open('a') as log:
    print('$',' '.join(cmd),flush=True);p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:snapshot();time.sleep(2)
    snapshot()
if p.returncode:
    print((dst/'drive.log').read_text()[-2500:],file=sys.stderr);sys.exit(p.returncode)
m=json.loads(manifest.read_text());progress=m['progress'];assert abs(progress['clip_seconds']-progress['moving_seconds'])<.002 and progress['full_windows']==progress['windows'];release=(root/'scene/hw4-cuda-224-254').resolve(strict=True);pub=json.loads((base/'progress-public/work/run/published-full.json').read_text());assert str(release)==pub['directory'];emit('drive_release',stage='final',directory=str(release),sha=pub['content_sha256']);emit('coverage',**progress);(dst/'manifest.json').write_text(json.dumps(m));(dst/'final.json').write_text(json.dumps(pub))
