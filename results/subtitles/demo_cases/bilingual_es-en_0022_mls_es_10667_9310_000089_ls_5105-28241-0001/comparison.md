# Switch-Point Demo: `bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001`

## Reference Ground Truth

**[ES]**  
por dar las señas de la taberna donde había estado aquella noche por las pausas que hacía hablando se hubiera podido creer que el caballero tomaba nota de dichas señas cuando le hubo explicado las circunstancias del sitio desde donde podía mirarse excitar la atención

**─── ES→EN SWITCH at ~18.5s ───**

**[EN]**  
AFTER AN APPRENTICESHIP ON A MERCHANT SHIP HE HAD ENTERED THE IMPERIAL NAVY AND HAD ALREADY REACHED THE RANK OF LIEUTENANT WHEN THE COUNT APPOINTED HIM TO THE CHARGE OF HIS OWN PRIVATE YACHT IN WHICH HE WAS ACCUSTOMED TO SPEND BY FAR THE GREATER PART OF HIS TIME THROUGHOUT THE WINTER GENERALLY CRUISING IN THE MEDITERRANEAN WHILST IN THE SUMMER HE VISITED MORE NORTHERN WATERS

---

## Model Outputs

| Model | Transcript (condensed) |
|---|---|
| **Reference** | por dar las señas de la taberna donde había estado aquella noche por las pausas que hacía hablando se hubiera podido creer que el caballero … |
| **Whisper** | _not available_ |
| **WhisperX** | _not available_ |

---

## Files in This Directory

| File | Status |
|---|---|
| `reference.srt` | ✓ |
| `reference.txt` | ✓ |
| `whisper.srt` | ✗ not available |
| `whisperx.srt` | ✗ not available |
| `comparison.md` | ✓ (this file) |

---

## Generate `whisper.srt`

```bash
python scripts/export_subtitles.py \
    --manifest   data/manifests/bilingual_es-en_50.jsonl \
    --sample-id  bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001 \
    --output-dir results\subtitles\demo_cases\bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001 \
    --model      large-v3 \
    --translate-es \
    --subtitle-mode bilingual
# rename: mv "results\subtitles\demo_cases\bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001\bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001.srt" "results\subtitles\demo_cases\bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001\whisper.srt"
```
