# Reproduction pipeline. Each stage is cached: delete its output to force a re-run.
OUT ?= outputs
DATA ?= data
SEED ?= 42

.PHONY: all figure verify test smoke clean-artifacts

all: figure

$(OUT)/motifs.npz:
	uv run sven-prep-data --data-dir $(DATA) --out $@

$(OUT)/of_encoder.npz: $(OUT)/motifs.npz
	uv run sven-train-encoder --motifs $< --out $@ --seed $(SEED)

$(OUT)/of_ven_model_k4max.npz: $(OUT)/of_encoder.npz $(OUT)/motifs.npz
	uv run sven-train-ven --encoder $(OUT)/of_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $@ --seed $(SEED) \
		--drive-e 0.06 --b-scale 16 --tau-s 20 --alpha-theta 0 --n-rend 600

## Regenerate the cancellation figure (runs the whole chain if needed).
figure: $(OUT)/of_ven_model_k4max.npz
	uv run sven-figure --model $< --encoder $(OUT)/of_encoder.npz \
		--motifs $(OUT)/motifs.npz --out-dir $(OUT)

## Verify downloaded + generated artifacts against the recorded checksums.
verify:
	uv run python -m spiking_ven.manifest verify --manifest MANIFEST.sha256 --root .

## Record checksums for the current artifacts.
manifest:
	uv run python -m spiking_ven.manifest write --manifest MANIFEST.sha256 --root .

test:
	uv run pytest -m "not slow"

## Fast end-to-end smoke run (tiny encoder + few renditions) — what CI runs.
smoke:
	uv run sven-train-encoder --motifs $(OUT)/motifs.npz --out $(OUT)/smoke_encoder.npz \
		--n-bases 16 --n-epochs 5 --seed $(SEED)
	uv run sven-train-ven --encoder $(OUT)/smoke_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $(OUT)/smoke_ven.npz --n-rend 20 --seed $(SEED)

clean-artifacts:
	rm -f $(OUT)/of_encoder.npz $(OUT)/of_ven_model_k4max.npz $(OUT)/encoding_comparison_k4max.*
