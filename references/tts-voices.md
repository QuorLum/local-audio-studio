# Speech with Kokoro-82M (`--mode tts`)

Read this when you choose a voice or language, prepare a script, fix a mispronunciation, or tune pacing.

**Contents** - [Voices](#voices) - [Languages](#languages) - [Preparing the text](#preparing-the-text) - [Pronunciation control](#pronunciation-control) - [Pacing](#pacing-speed-chunks-pauses) - [Blending voices](#blending-voices) - [Limits](#limits)

## Voices

Kokoro v1.0 ships 54 voices. `python scripts/generate_audio.py --list-voices` shows which are installed. Names are `<language><gender>_<name>`: `a`/`b` = American/British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Brazilian Portuguese, `z` Mandarin; `f`/`m` = female/male.

The author's grade estimates the **quality and quantity of each voice's training data**, not how pleasant it sounds to you; judge by ear. Source: Kokoro's `VOICES.md`.

| Language | Best-rated voices (grade) | All voices |
|---|---|---|
| American English | `af_heart` **A** (default), `af_bella` A-, `af_nicole` B-, then C+: `af_kore`, `af_sarah`, `af_aoede`, `am_fenrir`, `am_michael`, `am_puck` | af: alloy aoede bella heart jessica kore nicole nova river sarah sky - am: adam echo eric fenrir liam michael onyx puck santa |
| British English | `bf_emma` B-, `bf_isabella` C, `bm_fable` C, `bm_george` C | bf: alice emma isabella lily - bm: daniel fable george lewis |
| French | `ff_siwis` B- | ff_siwis |
| Hindi | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` (all C) | hf: alpha beta - hm: omega psi |
| Italian | `if_sara`, `im_nicola` (C) | if_sara, im_nicola |
| Japanese | `jf_alpha` C+ | jf: alpha gongitsune nezumi tebukuro - jm_kumo |
| Mandarin | all D | zf: xiaobei xiaoni xiaoxiao xiaoyi - zm: yunjian yunxi yunxia yunyang |
| Spanish / Portuguese (BR) | not graded | ef_dora em_alex em_santa - pf_dora pm_alex pm_santa |

Rule of thumb: for English narration start with `af_heart` (warm, clear) or `af_bella`; for a British read `bf_emma`. Lower-graded voices can be fine for short lines but are more likely to mumble or rush.

## Languages

The language code is the first letter of the voice name and is picked automatically (`--lang` overrides).

* English, Spanish, French, Hindi, Italian, Portuguese: work after the normal install. The espeak-ng library that handles the non-English text front-end comes bundled in a Python wheel (`espeakng-loader`).
* Japanese needs `pip install "misaki[ja]"`; Mandarin needs `pip install "misaki[zh]"`.
* Non-English support is thinner than English (less training data, weaker text-to-phoneme rules). Listen before committing to a long script.

## Preparing the text

* Write it the way it should be **spoken**. Kokoro's front-end already expands years ("1999" -> nineteen ninety-nine), ordinals (3rd), currency (`$5.99`), percentages and common abbreviations (Dr., Mr., etc.). It does **not** read these well, so spell them out: clock times ("3:45" -> "three forty-five"), dates ("12/05/2024" -> "December fifth, twenty twenty-four"), phone numbers (digit by digit) and symbols other than `%` and `$`.
* Acronyms: ALL CAPS words are read as ordinary words ("NASA" -> "nassa"). For letter-by-letter reading write "F B I" or use the phoneme markup below.
* Punctuation is pacing: commas give short breaths, full stops end the phrase, and question marks tend to lift the pitch. Avoid long run-on sentences.
* Separate paragraphs with a blank line - the tool leaves a longer pause there (`--paragraph-pause`, default 0.6 s; between sentences `--sentence-pause`, 0.18 s).
* One sentence per line without final punctuation also works: every line becomes a sentence.
* For long scripts use `--text-file script.txt` (or `-` for stdin). Nothing needs to be split by hand.

## Pronunciation control

Kokoro's text front-end (misaki) understands inline markup:

* Custom phonemes: `[Kokoro](/kˈOkəɹO/)` speaks the word with exactly those phonemes.
* Stress: `[word](+2)` raises and `[word](-2)` lowers the stress of one word.

Use it for brand names, acronyms and foreign words. When a name is mangled, try a respelling first ("Nguyen" -> "win"), then phonemes.

## Pacing: speed, chunks, pauses

* `--speed` 0.5-2.0 (default 1.0). Slow to 0.9 for dense text; 1.1 for energetic promos.
* Kokoro is most reliable on **100-200 phoneme tokens** per call: it is weak on very short utterances and rushes on very long ones. The tool therefore packs sentences into chunks of at most `--max-chars` characters (default 240), never cuts inside a sentence, and splits an over-long sentence at clause punctuation. Lower `--max-chars` if a voice rushes; raise it up to 400 for slower, steadier voices.
* Very short standalone lines ("Yes.", "Next.") sound weak on their own; put them next to a neighbouring sentence in the same paragraph.

## Blending voices

`--voice af_heart:0.7,af_bella:0.3` mixes voice embeddings with weights. Blends are free and often smooth out a single voice's quirks. Keep voices of the same language together.

## Limits

* No voice cloning, no emotion tags, no streaming, no SSML. Speaking style comes from the voice, the punctuation and `--speed`.
* Output is 24 kHz mono; use `--sample-rate 48000` for video editors (this resamples, it does not add detail).
* Everything runs in fp32 on CPU or GPU. The model is small (82M parameters): in our tests a *single* CPU core of a cloud VM synthesized speech at about 1.3x real time, so a GPU is optional for speech.
