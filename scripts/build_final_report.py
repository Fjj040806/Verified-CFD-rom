"""Compile the academic manuscript from editable LaTeX source."""
from pathlib import Path
import shutil,subprocess,tempfile,argparse
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--auxiliary-directory',type=Path)
args=parser.parse_args()
compiler=shutil.which('pdflatex')
if not compiler:raise SystemExit('Install a LaTeX distribution providing pdflatex to rebuild the manuscript.')
with tempfile.TemporaryDirectory() as directory:
 for _ in range(3):
  result=subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','-output-directory='+directory,'final-report.tex'],cwd=root/'reports',capture_output=True,text=True)
  if result.returncode:raise SystemExit(result.stdout[-5000:])
 shutil.copyfile(Path(directory)/'final-report.pdf',root/'reports/final-report.pdf')
 if args.auxiliary_directory:
  args.auxiliary_directory.mkdir(parents=True,exist_ok=True)
  for suffix in ['aux','log']:
   shutil.copyfile(Path(directory)/('final-report.'+suffix),args.auxiliary_directory/('final-report.'+suffix))
print(root/'reports/final-report.pdf')
