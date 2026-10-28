"""Hand-built lookup tables (plan 5.3, 5.4): pipeline_tag -> coarse modality, major organisations.

Written to data/external/{modality_map,major_orgs}.csv by scripts/p1_audit.py so the paper
appendix can list them in full.
"""
MODALITY = {
    "text": ["text-generation", "text-classification", "token-classification", "fill-mask", "feature-extraction",
             "question-answering", "sentence-similarity", "translation", "summarization", "text2text-generation",
             "multiple-choice", "zero-shot-classification", "text-ranking", "table-question-answering",
             "text-retrieval", "table-to-text", "conversational"],
    "image_gen": ["text-to-image", "unconditional-image-generation", "image-to-image", "text-to-video",
                  "image-to-video", "video-to-video", "text-to-3d", "image-to-3d"],
    "vision": ["image-classification", "object-detection", "image-segmentation", "video-classification",
               "zero-shot-image-classification", "image-feature-extraction", "depth-estimation", "mask-generation",
               "zero-shot-object-detection", "keypoint-detection"],
    "speech": ["automatic-speech-recognition", "audio-to-audio", "audio-classification", "text-to-speech",
               "text-to-audio", "voice-activity-detection"],
    "multimodal": ["image-to-text", "any-to-any", "image-text-to-text", "visual-question-answering",
                   "document-question-answering", "video-text-to-text", "audio-text-to-text",
                   "visual-document-retrieval"],
    "other": ["reinforcement-learning", "robotics", "tabular-classification", "tabular-regression",
              "time-series-forecasting", "graph-ml", "other"],
}
MODALITIES = list(MODALITY) + ["unknown"]
TAG2MOD = {t: m for m, ts in MODALITY.items() for t in ts}
LIB2MOD = {"diffusers": "image_gen"}  # used only when neither own tag nor parent gives a modality

# organisations that release base models (plan 5.4); redistributors such as unsloth/TheBloke excluded
MAJOR_ORGS = [
    "meta-llama", "facebook", "FacebookAI", "Qwen", "mistralai", "google", "google-bert", "google-t5", "microsoft",
    "deepseek-ai", "black-forest-labs", "stabilityai", "stable-diffusion-v1-5", "runwayml", "CompVis", "openai",
    "openai-community", "distilbert", "nvidia", "ibm-granite", "ibm", "tiiuae", "EleutherAI", "bigscience",
    "allenai", "HuggingFaceTB", "HuggingFaceH4", "HuggingFaceM4", "apple", "amazon", "Salesforce", "THUDM",
    "zai-org", "01-ai", "baichuan-inc", "internlm", "OpenGVLab", "xai-org", "CohereForAI", "CohereLabs",
    "ai21labs", "moonshotai", "BAAI", "sentence-transformers", "intfloat", "Alibaba-NLP", "laion", "mosaicml",
    "databricks", "openbmb", "bigcode", "Snowflake", "jinaai", "nomic-ai", "lmsys", "mistral-community",
    "stabilityai", "Wan-AI", "tencent", "ByteDance", "bytedance-research", "lllyasviel", "xinsir",
]
