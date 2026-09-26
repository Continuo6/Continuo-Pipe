# Example input

`manifest.example.jsonl` shows the three shapes a manifest row can take:

1. **Everything supplied** — `id`, a relative `wav_path`, `txt` and `lang`. This is the
   fast path: with `txt` on *every* row the annotate pass derives speaking rate from it
   and never loads whisper at all.
2. **Same, in English** — caption language is drawn independently of the audio's
   language, so an English clip may still receive a Chinese instruction.
3. **Bare minimum** — only `wav_path`. `id` defaults to the file's basename, and the
   language comes from whisper.

Relative paths resolve against `--audio-root` when you pass one (and are then confined
to it), otherwise against the working directory.

Try it against your own audio:

```bash
continuo-annotate --manifest examples/manifest.example.jsonl --audio-root corpus --limit 5
```

Mixing rows with and without `txt` is allowed, but a single row missing it forces the
whisper load for the whole run — so fill it in everywhere or nowhere.
