## Data Pre-processing

### Convert from MusicXML

- Navigate to the data folder ```cd data/```
- Modify the ```ORI_FOLDER``` and ```DES_FOLDER``` in ```1_batch_xml2abc.py```, then run this script:
  ```
  python 1_batch_xml2abc.py
  ```
  This will conver the MusicXML files into standard ABC notation files.
- Modify the ```ORI_FOLDER```, ```INTERLEAVED_FOLDER```, ```AUGMENTED_FOLDER```, and ```EVAL_SPLIT``` in ```2_data_preprocess.py```:
  
  ```python
  ORI_FOLDER = ''  # Folder containing standard ABC notation files
  INTERLEAVED_FOLDER = ''   # Output interleaved ABC notation files that are compatible with CLaMP 2 to this folder
  AUGMENTED_FOLDER = ''   # On the basis of interleaved ABC, output key-augmented and rest-omitted files that are compatible with NotaGen to this folder
  EVAL_SPLIT = 0.1    # Evaluation data ratio
  ```
  then run this script:
  ```
  python 2_data_preprocess.py
  ```
  - The script will convert the standard ABC to interleaved ABC, which is compatible with CLaMP 2. The files will be under ```INTERLEAVED_FOLDER```.

  - This script will make 15 key signature folders under the ```AUGMENTED_FOLDER```, and output interleaved ABC notation files with rest bars omitted. This is the data representation that NotaGen adopts.
  
  - This script will also generate data index files for training NotaGen. It will randomly split train and eval sets according to the proportion ```EVAL_SPLIT``` defines. The index files will be named as ```{AUGMENTED_FOLDER}_train.jsonl``` and ```{AUGMENTED_FOLDER}_eval.jsonl```.

  - **Multi-length motifs**: for every length in ```MOTIF_LENGTHS``` (4-10), the script emits a separate real datapoint per piece and key -- ```<name>_len<L>_<key>.abc``` -- with the same tunebody but the best motif of exactly ```L``` (distinct) notes in the ```%motif``` header. The train/eval split happens at the *piece* level before the per-length expansion, so all 7 length-variants of a piece land in the same split. ```data/generate_synthetic_motifs.py``` mirrors this: ```TOP_N_MOTIFS``` (3) crops per piece for *each* length (21 crops/piece), written to ```synthetic_motifs_multilen_v1``` on scratch.

  - **Rhythmic motifs** (```motif/rhythm.py```): every length-```L``` header also carries the best *rhythmic* motif of exactly ```L``` notes, found with the same sliding-window count but over note durations. Each duration is written as an exact fraction of the beat, where the beat is ```1/denominator``` of the piece's *initial* ```M:``` field (4/4 -> quarter, 6/8 -> eighth; with no ```M:``` the ```L:``` unit is the beat). Rests are events with a ```z``` suffix (```1z``` = quarter rest in 4/4): they sit inside a window but do not count toward ```L```, and a rest before the first or after the last note is outside the window. Ties merge, chords are one event, grace notes are dropped, tuplets and broken rhythms are resolved; there is no repeated-value collapsing and no inversion. The block is ```%motif:v1:rhythm: <ratios>``` next to the melodic pattern line, then ```%motif:rhythm:count:``` / ```%motif:rhythm:abc:``` after the melodic count/realization. A length is emitted with both motifs or neither. Hand-checked cases: ```python data/test_rhythm_motif.py```.

### Add the Irishman dataset (V:1 / melody)

The [Irishman](https://huggingface.co/datasets/sander-wood/irishman) dataset (~216k Irish
folk tunes) is merged into the existing V:1 (melody) training set. The tunes are
*monophonic* ABC with no `%%score`/`V:` structure, so they are first wrapped into the
multi-voice shape the pipeline expects (see ```irishman_to_abc.py```), then run through the
same ```2_data_preprocess.py``` with **env overrides** so the output lands on scratch
(```/usr/xtmp```, to stay under the home quota) while its index entries are **appended** to
the Lieder ```abcfiles_processed_v1*.jsonl``` files (one mixed melody set).

Everything is wired in ```scripts/run_preprocess_irishman.sh``` (download → convert → merge):

```
sbatch scripts/run_preprocess_irishman.sh     # or: bash scripts/run_preprocess_irishman.sh
```

- Cap the amount merged via ```MAX_TUNES``` in ```irishman_to_abc.py``` (default 10000; the
  full 216k would be ~145× the Lieder set and ~38h to preprocess).
- The relevant ```2_data_preprocess.py``` overrides (all default to the original Lieder
  behavior when unset): ```PREP_ORI_FOLDER```, ```PREP_AUGMENTED_FOLDER```,
  ```PREP_INTERLEAVED_FOLDER```, ```PREP_APPEND_V1```, ```PREP_INDEX_MODE``` (```append```),
  ```PREP_{MAIN,TRAIN,EVAL}_INDEX```, and ```PREP_NUM_WORKERS``` (parallel preprocessing).

## Data Post-processing

### Preview Sheets in ABC Notation

We recommend [EasyABC](https://sourceforge.net/projects/easyabc/), a nice software for ABC Notation previewing, composing and editing.

It's needed to add a line "X:1" before each piece to present the score image in EasyABC :D

### Convert to MusicXML

- Go to the data folder ```cd data/```
- Modify the ```ORI_FOLDER``` and ```DES_FOLDER``` in ```3_batch_abc2xml.py```, then run this script:
  ```
  python 3_batch_abc2xml.py
  ```
  This will conver the standard/interleaved ABC notation files into MusicXML files.
