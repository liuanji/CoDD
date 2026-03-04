from .model_class import PyJuiceHubModel, CoDD, CoDDConfig, CoDDOutput, apply_pc_logits

# Re-export the model-specific subclasses when available.
# These live alongside their base model files in eval/ and require
# those packages to be importable.
try:
    from eval.llada.codd_llada import CoddLlada
except ImportError:
    try:
        from llada.codd_llada import CoddLlada
    except ImportError:
        pass

try:
    from eval.dream.codd_dream import CoddDream
except ImportError:
    try:
        from dream.codd_dream import CoddDream
    except ImportError:
        pass
