"""Create full and GitHub-ready archives; output outside the project tree."""
from pathlib import Path
import argparse,shutil,json,hashlib
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=root.parent/'release');p.add_argument('--stage-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
selected_figures={'phase4b-vortex-shedding.gif','phase4b-vortex-shedding.png','phase4b-diagnostics.png','phase5-rom.png','phase5-flow-comparison.png','phase6-controls.png','phase1-verification.png','phase2-verification.png','phase3-cavity.png','phase4-benchmark.png'}
def eligible(path):return '__pycache__' not in path.parts and '.git' not in path.parts and 'demo-output' not in path.parts and not path.name.endswith('.log') and path.name!='RELEASE-MANIFEST.json'
def light(rel):
 parts=rel.parts
 if parts[0] in {'src','tests','scripts','experiments','reports','data'}:return True
 if len(parts)==1:return True
 if parts[0]=='figures':return rel.name in selected_figures
 if parts[0]=='results':
  return rel.suffix=='.json' or rel.as_posix() in {'results/phase4b/coarse-half-velocity-mass.npz','results/phase4b/coarse-half-divergence.npz','results/phase4b/coarse-half-mesh.npz','results/phase5/models-and-reconstructions.npz'}
 return False
def manifest(base,kind):
 files=[]
 for f in sorted(base.rglob('*')):
  if f.is_file() and eligible(f.relative_to(base)):
   files.append(dict(path=f.relative_to(base).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 (base/'RELEASE-MANIFEST.json').write_text(json.dumps(dict(version='1.0',package=kind,files=files),indent=2))
stage=a.output/'github-ready'/root.name
if stage.exists():shutil.rmtree(stage)
stage.mkdir(parents=True)
for f in sorted(root.rglob('*')):
 if f.is_file() and eligible(f.relative_to(root)) and light(f.relative_to(root)):
  target=stage/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
manifest(stage,'github-ready')
if not a.stage_only:
 manifest(root,'full-research-archive')
 for base,name in [(root,'incompressible-flow-rom-final.zip'),(stage,'incompressible-flow-rom-github-ready.zip')]:
  archive=a.output/name
  with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
   for f in sorted(base.rglob('*')):
    if f.is_file() and ('RELEASE-MANIFEST.json'==f.name or eligible(f.relative_to(base))):z.write(f,(Path(root.name)/f.relative_to(base)).as_posix())
  print(name,archive.stat().st_size)
else:print(stage)
