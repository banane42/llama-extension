#pragma once
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <thread>
#include <atomic>

#include "llama_sampler_chain.h"

struct llama_model;
struct llama_context;

namespace godot {

class LlamaServer : public Object {
    GDCLASS(LlamaServer, Object)

private:
    // ── Singleton ─────────────────────────────────────────────────────────────
    static LlamaServer *_singleton;

    // ── llama.cpp state ───────────────────────────────────────────────────────
    llama_model   *_model   = nullptr;
    llama_context *_ctx     = nullptr;
    std::thread    _worker;
    std::atomic<bool> _running{false};

    // ── Config ────────────────────────────────────────────────────────────────
    String _model_path;
    int    _max_tokens = 256;

    // The server-level default sampler chain. Must be set explicitly before any
    // generation call that does not supply its own chain — no hidden fallback exists.
    Ref<LlamaSamplerChain> _default_sampler_chain;

    // System prompt — use {prompt} as the placeholder for user input.
    String _system_prompt =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "{prompt}";

    // ── Internal ──────────────────────────────────────────────────────────────

    // Resolve which chain to use for a call: prefer `override_chain` when valid,
    // fall back to `_default_sampler_chain`, or emit an error and return nullptr.
    // Returns a freshly-built llama_sampler* the caller must free, or nullptr on error.
    llama_sampler *_resolve_sampler_chain(const Ref<LlamaSamplerChain> &override_chain);

    // prompt is the raw user text; chain must be non-null (validated by callers).
    String _run_inference(const String &prompt, llama_sampler *chain);
    void   _do_generate(String prompt, Ref<LlamaSamplerChain> chain);
    String _build_prompt(const String &user_prompt) const;

protected:
    static void _bind_methods();

public:
    static LlamaServer *get_singleton();

    LlamaServer();
    ~LlamaServer();

    // ── Model lifecycle ───────────────────────────────────────────────────────

    /**
     * @brief Load a GGUF model from disk.
     * @param model_path  Path to the .gguf file.
     * @param n_gpu_layers Number of layers to offload to GPU. -1 = system default.
     * @param n_ctx        Context window size. 0 = model default.
     */
    bool load_model(String model_path, int n_gpu_layers = -1, int n_ctx = 0);
    void unload_model();
    bool is_model_loaded() const { return _model != nullptr; }

    // ── Generation ────────────────────────────────────────────────────────────

    /// Async generation. Emits generation_completed or generation_failed.
    /// Pass a LlamaSamplerChain to override the server default for this call only.
    void generate(String prompt,
                  Ref<LlamaSamplerChain> chain = Ref<LlamaSamplerChain>());

    /// Blocking generation. Returns empty string on error.
    /// Pass a LlamaSamplerChain to override the server default for this call only.
    String generate_sync(String prompt,
                         Ref<LlamaSamplerChain> chain = Ref<LlamaSamplerChain>());

    // ── Getters / Setters ─────────────────────────────────────────────────────
    void   set_model_path(String path)  { _model_path = path; }
    String get_model_path() const       { return _model_path; }

    void set_max_tokens(int n)          { _max_tokens = n; }
    int  get_max_tokens() const         { return _max_tokens; }

    /// The server-level default sampler chain. Must be set before calling
    /// generate() or generate_sync() without an explicit chain argument.
    void                   set_default_sampler_chain(Ref<LlamaSamplerChain> chain) { _default_sampler_chain = chain; }
    Ref<LlamaSamplerChain> get_default_sampler_chain() const                       { return _default_sampler_chain; }

    void   set_system_prompt(String p)  { _system_prompt = p; }
    String get_system_prompt() const    { return _system_prompt; }
};

} // namespace godot
