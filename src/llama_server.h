#pragma once
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <thread>
#include <atomic>

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
    int    _max_tokens   = 256;

    // Sampler settings
    int   _top_k        = 40;
    float _top_p        = 0.95f;
    float _temperature  = 0.7f;

    // Penalty settings
    int   _penalty_last_n   = 64;
    float _penalty_repeat   = 1.1f;

    // System prompt — use {prompt} as the placeholder for user input
    String _system_prompt =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "{prompt}";

    // ── Internal ──────────────────────────────────────────────────────────────
    String _run_inference(const String &prompt);
    void   _do_generate(String prompt);
    String _build_prompt(const String &user_prompt) const;

protected:
    static void _bind_methods();

public:
    static LlamaServer *get_singleton();

    LlamaServer();
    ~LlamaServer();

    // ── Model lifecycle ───────────────────────────────────────────────────────

    /**
     * @brief loads model
     * @param model_path path to the gguf model to be loaded
     * @param n_gpu_layer number of gpu layers to use, -1 to configure to system defaults
     * @param n_ctx Size of context window, 0 for model default
     */
    bool load_model(String model_path, int n_gpu_layers = -1, int n_ctx = 0);
    void unload_model();
    bool is_model_loaded() const { return _model != nullptr; }

    // ── Generation ────────────────────────────────────────────────────────────
    void   generate(String prompt);           // async — emits generation_completed
    String generate_sync(String prompt);      // blocking

    // ── Getters / Setters ─────────────────────────────────────────────────────
    void   set_model_path(String path)      { _model_path = path; }
    String get_model_path() const           { return _model_path; }

    void set_max_tokens(int n)              { _max_tokens = n; }
    int  get_max_tokens() const             { return _max_tokens; }

    void  set_top_k(int n)                  { _top_k = n; }
    int   get_top_k() const                 { return _top_k; }

    void  set_top_p(float p)                { _top_p = p; }
    float get_top_p() const                 { return _top_p; }

    void  set_temperature(float t)          { _temperature = t; }
    float get_temperature() const           { return _temperature; }

    void  set_penalty_last_n(int n)         { _penalty_last_n = n; }
    int   get_penalty_last_n() const        { return _penalty_last_n; }

    void  set_penalty_repeat(float p)       { _penalty_repeat = p; }
    float get_penalty_repeat() const        { return _penalty_repeat; }

    void   set_system_prompt(String p)      { _system_prompt = p; }
    String get_system_prompt() const        { return _system_prompt; }
};

} // namespace godot