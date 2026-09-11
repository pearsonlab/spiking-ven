# Reproduction pipeline. Each stage is cached: delete its output to force a re-run.
OUT  ?= outputs
DATA ?= data
SEED ?= 42

.PHONY: all figure rates verify manifest test smoke clean-artifacts

all: figure

## 1. Song corpus -> outputs/motifs.npz  (needs the `data` extra for soundfile)
$(OUT)/motifs.npz:
	uv run --extra data sven-prep-data --data-dir $(DATA) --out $@

## 2. Sparse encoder -> outputs/of_encoder.npz
$(OUT)/of_encoder.npz: $(OUT)/motifs.npz
	uv run sven-train-encoder --motifs $< --out $@ --seed $(SEED)

## 3. Vocal error network -> outputs/of_ven_model_k4max.npz
##    Flags are spelled out even though they are the defaults, so the recipe is
##    self-documenting: this is the tuned operating point.
$(OUT)/of_ven_model_k4max.npz: $(OUT)/of_encoder.npz $(OUT)/motifs.npz
	uv run sven-train-ven --encoder $(OUT)/of_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $@ --seed $(SEED) \
		--drive-e 0.06 --b-scale 16 --tau-s 20 --alpha-theta 0 --n-rend 600

## 4. The cancellation figure (runs the whole chain if needed).
figure: $(OUT)/of_ven_model_k4max.npz
	uv run --extra plots sven-figure --model $< --of-encoder $(OUT)/of_encoder.npz \
		--motifs $(OUT)/motifs.npz --out-dir $(OUT)

## Supplementary population-rate view.
rates: $(OUT)/of_ven_model_k4max.npz
	uv run --extra plots sven-figure --view rates --model $< \
		--of-encoder $(OUT)/of_encoder.npz \
		--motifs $(OUT)/motifs.npz --out-dir $(OUT)

## Verify the generated artifacts against the recorded checksums.
verify:
	uv run python -m spiking_ven.manifest verify --manifest MANIFEST.sha256 --root .

## Record checksums for the current artifacts.
manifest:
	uv run python -m spiking_ven.manifest write --manifest MANIFEST.sha256 --root .

test:
	uv run --extra dev pytest -m "not slow"

## The full reproduction assertions (minutes) -- needs outputs/ populated.
test-repro:
	uv run --extra dev --extra plots pytest -m slow -v

## Fast end-to-end smoke run (tiny encoder, few renditions) -- what CI runs.
smoke: $(OUT)/motifs.npz
	uv run sven-train-encoder --motifs $(OUT)/motifs.npz --out $(OUT)/smoke_encoder.npz \
		--n-bases 16 --n-epochs 5 --seed $(SEED)
	uv run sven-train-ven --encoder $(OUT)/smoke_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $(OUT)/smoke_ven.npz --n-rend 3 --seed $(SEED)

clean-artifacts:
	rm -f $(OUT)/of_encoder.npz $(OUT)/of_ven_model_k4max.npz \
		$(OUT)/encoding_comparison*.png $(OUT)/encoding_comparison*.pdf
