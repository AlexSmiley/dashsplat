"""Fetch exactly the revision used by the drive worker; disclose card and upstream terms before weights."""
from pathlib import Path
from huggingface_hub import hf_hub_download,snapshot_download
import re

MODELS = [
    ('facebook/mask2former-swin-large-cityscapes-panoptic','544d76fe93971ee046dacae19b6d4f6ecb5d9088','Mask2Former code MIT; Cityscapes non-commercial dataset terms'),
    ('depth-anything/DA3-LARGE-1.1','0e109ae307c5982f319a67cf6f9f99ccdc0ec97c','CC BY-NC 4.0'),
    ('depth-anything/DA3METRIC-LARGE','4010e39f3634a45bc60553321fb49fb760bd594e','Apache-2.0; short-baseline fallback'),
    ('google/siglip2-base-patch16-224','75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2','Apache-2.0'),
    ('nvidia/difix_ref','d4830559772a5795c9d136302c5b197d6418d3fb','NVIDIA research/non-commercial and SD-Turbo terms'),
]
for repo,rev,terms in MODELS:
    card=Path(hf_hub_download(repo,'README.md',revision=rev)).read_text();line=re.search(r'^license:.*$',card,re.M)
    print(f'{repo}@{rev}: {line.group() if line else "license: see model card"}; {terms}',flush=True)
    from huggingface_hub import list_repo_files
    files=list_repo_files(repo,revision=rev); has_safe=any(f.endswith('.safetensors') for f in files)
    p=Path(snapshot_download(repo,revision=rev,ignore_patterns=(['*.bin'] if has_safe else [])+['*.h5','*.msgpack','onnx/*','*.onnx','*.ot']));assert p.name==rev
    refs=p.parent.parent/'refs';refs.mkdir(exist_ok=True);(refs/'main').write_text(rev)
