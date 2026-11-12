# Examples

Do not commit personal photos to this repository.

Run the CLI from the repository root:

```bash
beauty-score score --input /path/to/photo.jpg --output-dir outputs/
```

Use the Python API:

```python
from beauty_scorer import BeautyScorer

scorer = BeautyScorer()
for result in scorer.score_file("/path/to/photo.jpg"):
    print(f"score={result.score:.2f} percentile={result.percentile:.1f}%")
```

Score a folder and read the JSON output:

```bash
beauty-score score --input /path/to/folder --json > results.json
```

Real-time camera:

```bash
beauty-score video --source 0
```
