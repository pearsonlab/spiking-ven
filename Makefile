# Reproduction pipeline. Each stage is cached: delete its output to force a re-run.
OUT  ?= outputs
DATA ?= data
SEED ?= 42

.PHONY: all figure psth verify manifest test test-repro smoke clean-artifacts

all: figure

## 1. Song corpus -> outputs/motifs.npz  (needs the `data` extra for soundfile)
$(OUT)/motifs.npz:
	uv run --extra data sven-prep-data --data-dir $(DATA) --out $@

## 2. Causal minimum-phase encoder -> outputs/causal_encoder.npz
$(OUT)/causal_encoder.npz: $(OUT)/motifs.npz
	uv run sven-train-encoder --motifs $< --out $@ --seed $(SEED)

## 3. Vocal error network -> outputs/of_ven_model_default.npz
##    No tuning flags: the operating point is the default, and lives in constants.py.
$(OUT)/of_ven_model_default.npz: $(OUT)/causal_encoder.npz $(OUT)/motifs.npz
	uv run sven-train-ven --encoder $(OUT)/causal_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $@ --seed $(SEED)

## 4. The cancellation figure (runs the whole chain if needed).
figure: $(OUT)/of_ven_model_default.npz
	uv run --extra plots sven-figure --model $< --encoder $(OUT)/causal_encoder.npz \
		--motifs $(OUT)/motifs.npz --out-dir $(OUT)

## The DAF response time course, against the Fig 8E reading.
psth: $(OUT)/of_ven_model_default.npz
	uv run --extra plots sven-figure --view psth --model $< \
		--encoder $(OUT)/causal_encoder.npz \
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

## Fast end-to-end smoke run (tiny encoder, few renditions).
smoke: $(OUT)/motifs.npz
	uv run sven-train-encoder --motifs $(OUT)/motifs.npz --out $(OUT)/smoke_encoder.npz \
		--n-bases 16 --causal-epochs 5 --seed $(SEED)
	uv run sven-train-ven --encoder $(OUT)/smoke_encoder.npz --motifs $(OUT)/motifs.npz \
		--out $(OUT)/smoke_ven.npz --n-rend 3 --seed $(SEED)

clean-artifacts:
	rm -f $(OUT)/causal_encoder.npz $(OUT)/of_ven_model_default.npz \
		$(OUT)/of_ven_model_default_traces.npz $(OUT)/daf_psth*.p* \
		$(OUT)/encoding_comparison*.png $(OUT)/encoding_comparison*.pdf
