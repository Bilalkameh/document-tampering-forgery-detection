# SIDTD Inspection

## Setup result

SIDTD was cloned into:

external/SIDTD_Dataset/

The package was installed in editable mode in the repo virtual environment.

Import check:

```bash
python -c "import SIDTD; print('SIDTD import OK')"
```

Result:

```text
SIDTD import OK
```

## Generator files

Files found by searching for generator-related filenames:

- external/SIDTD_Dataset/SIDTD/utils/batch_generator.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv500/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv500/Template_Generator.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/README.md
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/Video_Generator.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/Template_Generator.py

Files under `SIDTD/data/DataGenerator`:

- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv500/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv500/Template_Generator.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/README.md
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/__init__.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/Video_Generator.py
- external/SIDTD_Dataset/SIDTD/data/DataGenerator/Midv2020/Template_Generator.py

## Coordinate answer

SIDTD does not directly write `x1, y1, x2, y2` into generated fake metadata.

The generated metadata stores fields such as:

- `src`
- `second_src`
- `field`
- `second_field`
- `shift`
- `type_transformation`

However, SIDTD's source annotations do contain region coordinates. The annotation JSONs map each field name to a `quad`, for example:

```json
"name": {
  "value": "Adnan",
  "quad": [[704, 365], [880, 365], [880, 412], [704, 412]]
}
```

SIDTD also has utilities for converting those coordinates:

- `bbox_info(info)` in `SIDTD/utils/util.py`
- `bbox_to_coord(x, y, w, h)` in `SIDTD/utils/util.py`

## Practical path

This is between Case A and Case B:

- SIDTD does expose the changed field name in generated metadata.
- The exact region can be recovered by looking up that field in the source annotation JSON.
- For extra safety, especially with shifted crop/replace operations, we can also compute a pixel-difference bbox between the original and generated fake image.

Recommended approach for YOLO labels:

1. Generate a fake image with SIDTD.
2. Read `fake_meta["field"]`.
3. Load the original annotation JSON from `fake_meta["src"]`.
4. Convert that field's `quad` to `x1, y1, x2, y2`.
5. Optionally refine or validate the box using original-vs-fake pixel difference.
