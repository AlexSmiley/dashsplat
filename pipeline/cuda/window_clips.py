"""Trim on the trusted source host. Preserve native YUV, per-picture Tesla SEI, IDs and the master clock."""
import argparse, hashlib, json, struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import av
from pipeline.ingest.ingest import parse_packet,CAMS

def nals(packet):
    data=bytes(packet);pos=0
    while pos<len(data):
        size=struct.unpack_from('>I',data,pos)[0];pos+=4
        if size<=0 or pos+size>len(data):raise ValueError('invalid AVC sample')
        yield data[pos:pos+size];pos+=size

def trim(group,cam,clock,offset,lo,hi,dst):
    src=Path(group['paths'][cam]); out=dst/src.name; ledger=[]; pending={}; idx=group['starts'][cam]; mux=None
    with av.open(src) as f:
        st=f.streams.video[0];st.codec_context.thread_count=2
        for p in f.demux(st):
            if p.size:
                sei,pic=parse_packet(bytes(p));
                if not pic:continue
                pending[p.pts]=(sei,next((n for n in nals(p) if n[0]&31==6 and parse_packet(struct.pack('>I',len(n))+n)[0] is not None),None))
            for frame in p.decode():
                n=idx;idx+=1;sei,blob=pending.pop(frame.pts);seq=(sei or {}).get('seq')
                if seq is None:continue
                tm=(seq-clock['seq0'])/clock['fps'];physical=tm-offset/clock['fps']
                if not(lo<=tm<=hi and lo<=physical<=hi):continue
                if not blob:raise ValueError('missing Tesla SEI')
                if mux is None:
                    mux=av.open(out,'w',format='mp4');enc=mux.add_stream('libx264',rate=st.average_rate);enc.width=st.width;enc.height=st.height;enc.pix_fmt=frame.format.name;enc.time_base=st.time_base;enc.codec_context.time_base=st.time_base;enc.codec_context.thread_count=2;enc.options={'qp':'0','preset':'ultrafast','tune':'zerolatency','x264-params':'bframes=0:scenecut=0'}
                pix=hashlib.sha256(frame.to_ndarray(format='yuv420p')).digest();ps=enc.encode(frame);assert len(ps)==1;ep=ps[0];owned=av.Packet(bytes([0,0,0,1])+blob+bytes(ep));owned.pts=ep.pts;owned.dts=ep.dts;owned.time_base=ep.time_base;owned.stream=enc;owned.is_keyframe=ep.is_keyframe;mux.mux(owned);ledger.append((n,sei,pix))
    if not mux:return None
    assert not enc.encode(None);mux.close();seen=0;pending={}
    with av.open(out) as f:
        st=f.streams.video[0]
        for p in f.demux(st):
            if p.size:
                sei,picture=parse_packet(bytes(p));assert picture and sei;pending[p.pts]=sei
            for fr in p.decode():
                _,sei,digest=ledger[seen];assert pending.pop(fr.pts)==sei and hashlib.sha256(fr.to_ndarray(format='yuv420p')).digest()==digest;seen+=1
    assert seen==len(ledger) and not pending;print(src.name,seen,'verified pictures + SEI',flush=True);return [row[0] for row in ledger]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('inventory',type=Path);ap.add_argument('destination',type=Path);ap.add_argument('--window',nargs=2,type=float,required=True);a=ap.parse_args();lo,hi=a.window
    doc=json.loads(a.inventory.read_text());offs=json.loads((a.inventory.parent/'data/drive/meta.json').read_text())['seq_offsets'];dst=a.destination;dst.mkdir(mode=0o700,parents=True,exist_ok=False);ics={}
    with ThreadPoolExecutor(3) as pool:
        for g in doc['clips']:
            if (g['seq'][1]-doc['seq0'])/doc['fps']<lo-.5 or (g['seq'][0]-doc['seq0'])/doc['fps']>hi+.5:continue
            gs=dict(zip(CAMS,pool.map(lambda c:trim(g,c,doc,offs[c],lo,hi,dst),CAMS)));assert all(gs.values());ics[g['name']]=gs
    (dst/'window-clock.json').write_text(json.dumps(dict(version=1,window=a.window,seq0=doc['seq0'],fps=doc['fps'],native_ids=ics)));print('verified window',a.window,flush=True)
