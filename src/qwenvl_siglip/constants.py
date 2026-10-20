"""Constants shared by the model, data pipeline and evaluation."""

NUM_QUERIES = 256  # learnable queries in the resampler -> visual tokens per image

# `<imgpad>` is a placeholder: NUM_QUERIES copies sit between <img> and </img>, and their
# embeddings are replaced by the resampler output at forward time.
SPECIAL_TOKENS = [
    "<img>", "</img>", "<imgpad>",
    "<ref>", "</ref>",
    "<box>", "</box>",
    "<quad>", "</quad>",
]  # fmt: skip

IMG_STR = "<img>" + "<imgpad>" * NUM_QUERIES + "</img>"
CAPTION_PROMPT = "Generate the caption in English:"
