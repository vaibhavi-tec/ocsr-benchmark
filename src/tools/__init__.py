"""Registry of available OCSR tools. Imports are lazy so one tool's
dependencies don't have to load when benchmarking another."""


def build_tool(name: str, tool_cfg: dict, bench_cfg: dict):
    common = dict(models_dir=bench_cfg["models_dir"], device=bench_cfg["device"],
                  batch_size=bench_cfg["batch_size"],
                  hf_repo=tool_cfg["hf_repo"], hf_file=tool_cfg["hf_file"])
    if name == "molscribe":
        from .molscribe_tool import MolScribeTool
        return MolScribeTool(**common)
    if name == "molglyph":
        from .molglyph_tool import MolGlyphTool
        return MolGlyphTool(**common)
    raise ValueError(f"Unknown tool '{name}'. Available: molscribe, molglyph")
