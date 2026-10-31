from pathlib import Path
p=Path('/content/tc-library-run')
print('EXIT:',(p/'exit-code.txt').read_text() if (p/'exit-code.txt').exists() else 'RUNNING')
print((p/'run.log').read_text()[-5000:] if (p/'run.log').exists() else 'No log')
