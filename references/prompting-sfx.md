# Prompting sound effects & ambience (`--mode sfx`)

Read this when you write or review an SFX prompt, or when a generated effect sounds synthetic, vague or wrong.

**Contents** - [How the model reads a prompt](#how-the-model-reads-a-prompt) - [The three ingredients](#the-three-ingredients) - [What the built-in enricher adds](#what-the-built-in-enricher-adds) - [Before / after](#before--after) - [Duration cheat sheet](#duration-cheat-sheet) - [Beds, loops and long ambience](#beds-loops-and-long-ambience) - [Steering](#steering-seeds-cfg-negative-prompts) - [Common mistakes](#common-mistakes)

## How the model reads a prompt

Stable Audio 3 was trained on recordings from Freesound and AudioSparx **together with their metadata**. A prompt works best when it reads like the description a sound librarian would write for a *real recording*: what made the sound, how it was triggered, where the microphone was and what the room did to it. A bare noun ("rain") lands in the middle of everything the model has heard under that word and tends to sound like generic synthetic audio.

The tag `TrackType: SFX` at the start tells the model it is making a sound effect or sample; Stability reports it produces more semantically reasonable results. The CLI adds it by default (`--no-tags` removes it).

## The three ingredients

Write every effect prompt as **source + action + production**:

| Ingredient | Question to answer | Examples |
|---|---|---|
| **Source** | What object, creature or machine makes the sound? | "a heavy oak door", "14-inch hi-hats", "a pack of wolves", "a diesel truck" |
| **Action** | How is it triggered and how does it end? | "slamming shut with a fast decay", "creaking open slowly", "idling, then revving once" |
| **Production** | Where is the mic, what is the room like, how is it processed? | "close-mic foley recording, dry studio acoustics", "binaural outdoor field recording", "dark wooden reverb", "touch of analog saturation" |

Template: `TrackType: SFX, <source> <action>, <mic/perspective>, <room>, <texture>, no music`

The built-in enricher can only add the *production* ingredient (it cannot guess your source or action), so spend your own words on source and action.

## What the built-in enricher adds

`generate_audio.py --mode sfx` classifies the prompt by keywords and fills only the production slots the prompt does not already cover. A slot is skipped when you already mention it (for example "cathedral" or "reverb" suppresses the room slot, "lo-fi" or "crisp" the texture slot). `no music` is appended unless the prompt itself mentions music.

| category | typical keywords | mic perspective | room | texture |
|---|---|---|---|---|
| `ui_synth` | laser, zap, beep, blip, glitch, interface, sci-fi, riser | clean designed sound effect with precise stereo placement | dry, no room reverb | crisp high-fidelity studio sound design |
| `magic_fantasy` | magic, spell, shimmer, sparkle, fantasy, portal | layered sound design with wide stereo movement | airy natural reverb tail | clean high-fidelity |
| `impact_cinematic` | explosion, boom, impact, crash, slam, punch, whoosh, sword | close-mic'd cinematic sound effect | natural room tail | tight punchy transient with full low end |
| `foley` | footsteps, door, knock, creak, cloth, keys, typing, pour, glass, cooking | close-mic foley recording | dry studio acoustics | clean and detailed, no background noise |
| `ambience_nature` | rain, wind, forest, river, waterfall, ocean, storm, crickets, campfire | binaural outdoor field recording | open-air natural acoustics | wide stereo ambience with natural dynamics |
| `ambience_urban` | city, traffic, street, crowd, cafe, restaurant, station, office | stereo field recording on location | natural reverberant room tone | continuous ambient bed |
| `vehicle_machine` | car, engine, motorcycle, train, helicopter, fan, drill, siren | close-mic'd mechanical recording | in-situ acoustics | detailed and realistic |
| `animal` | dog, cat, lion, horse, rooster, wolf, owl, bark, roar | close-mic wildlife recording | natural outdoor acoustics | clean and realistic |
| `human_vocal` | laugh, cough, sneeze, scream, gasp, breath, applause | close-mic human vocalization | dry booth acoustics | clean and natural |
| `general` | (anything else) | clean high-fidelity recording | natural acoustics | detailed stereo image |

Override any slot with `--mic`, `--room`, `--texture`. Preview the result without loading a model: `--dry-run`.

## Before / after

| You write | Final prompt sent to the model |
|---|---|
| `footsteps on wet gravel` | `TrackType: SFX, footsteps on wet gravel, close-mic foley recording, dry studio acoustics, clean and detailed, no background noise, no music` |
| `rain on a tin roof` | `TrackType: SFX, rain on a tin roof, binaural outdoor field recording, open-air natural acoustics, wide stereo ambience with natural dynamics, no music` |
| `laser blast` | `TrackType: SFX, laser blast, clean designed sound effect with precise stereo placement, dry, no room reverb, crisp high-fidelity studio sound design, no music` |
| `a door creaking open in a large cathedral` | room slot skipped (you already described it): `TrackType: SFX, a door creaking open in a large cathedral, close-mic foley recording, clean and detailed, no background noise, no music` |
| *agent-written, run with `--no-enrich`* | `TrackType: SFX, a heavy oak door slamming shut with a fast decay and a low wooden thud, close-mic'd in a dead room, a touch of analog saturation, no music` |

## Duration cheat sheet

Pick a duration that fits what you describe; the model plans the whole clip for the length you ask, so a one-shot stretched to 30 s gets padded with tails and artifacts.

| Kind of sound | `--duration` |
|---|---|
| one-shot: click, hit, slam, pop, UI blip | 0.5 - 3 s |
| short sequence: 4-8 footsteps, door open/close, glass break | 3 - 8 s |
| riser, whoosh, transition, stinger | 2 - 6 s |
| animal call, vehicle pass-by, weather event | 4 - 15 s |
| ambience bed (rain, forest, cafe, room tone) | 30 - 120 s |
| longer than 120 s | continuation or looping, see below |

`--trim-silence` removes leading/trailing silence if a one-shot comes out padded.

## Beds, loops and long ambience

* **Seamless loop:** add `--loopable`. The tail is blended into the head and the file gets shorter by `--crossfade` seconds (default 2). Use it for game engines and video beds.
* **Longer than 120 s:** the tool continues the audio automatically (`--extend-method continue`: the last `--context` seconds, default 30, are fed back and only the new part is generated). For stationary textures such as rain or room tone, `--extend-method loop` (crossfaded looping of one 120 s clip) is cheaper and just as good. Each continuation step costs about one full 120 s generation.
* An ambience should describe a *steady state* ("steady rain", "constant low hum"). Events inside a long bed ("a car passes", "thunder") will repeat when looped; layer those as separate one-shots instead.

## Steering: seeds, CFG, negative prompts

* The seed is printed and returned in `--json`. Re-run with `--seed N` to reproduce a take, or try 3-4 seeds and keep the best one - cheap for short effects.
* The Small models are tuned for `--steps 8` with `--cfg-scale 1.0` (fast, no guidance). `--steps` accepts 1-12; more steps rarely help.
* `--cfg-scale 3 --negative-prompt "speech, music, reverb"` makes the model actively avoid those things, at roughly twice the compute. Words like "no music" in the *positive* prompt are only a weak hint for a text encoder, so use the negative prompt when it matters.

## Common mistakes

* **Several sounds in one prompt** ("dog barking while it rains and a car passes") blend into mush. Generate layers separately and mix them.
* **Speech inside an effect.** The SFX model does not produce intelligible words. Use `--mode tts` and mix.
* **Abstract adjectives only** ("epic", "cool sound"). Name the physical source and the action.
* **Over-long prompts.** The text encoder reads at most 256 tokens; the enricher keeps prompts under about 700 characters and drops the texture, then room, slot first when over budget.
* **Contradictions** ("dry studio" + "large cathedral"). If you write your own room, the enricher stays out of the way; if you pass `--room`, it wins.
