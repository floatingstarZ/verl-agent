# Shared H100 flash-attn runtime setup for GraphGPO recipes.

graphgpo_setup_flash_attn_sm90() {
    local default_site="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache/flash_attn_sm80_sm90_site"
    if [ ! -d "$default_site" ] && [ -d "/tmp/flash_attn_sm80_sm90_site" ]; then
        default_site="/tmp/flash_attn_sm80_sm90_site"
    fi
    local site="${FLASH_ATTN_SM90_SITE:-$default_site}"
    local ext
    ext="$(find "$site" -maxdepth 1 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1 || true)"
    if [ -z "$ext" ]; then
        echo "[FATAL] Missing sm_90 flash-attn override under $site" >&2
        echo "[FATAL] Set FLASH_ATTN_SM90_SITE or override actor_rollout_ref.model.attn_implementation=sdpa." >&2
        exit 1
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
        if ! cuobjdump --list-elf "$ext" 2>/dev/null | awk '/sm_90/{found=1} END{exit !found}'; then
            echo "[FATAL] $ext does not contain sm_90 kernels" >&2
            echo "[FATAL] Set FLASH_ATTN_SM90_SITE to an sm_90 build or override actor_rollout_ref.model.attn_implementation=sdpa." >&2
            exit 1
        fi
    fi
    export FLASH_ATTN_SM90_SITE="$site"
    export PYTHONPATH="$site:${PYTHONPATH:-}"
    echo "[INFO] Using sm_90 flash-attn from $site"
}
