#pragma once
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <thread>
#include <atomic>

// Forward declare to avoid pulling all of llama.h into the header
struct llama_model;
struct llama_context;

namespace godot {

class LlamaNode : public Node {
    GDCLASS(LlamaNode, Node)

private:
    llama_model   *_model   = nullptr;
    llama_context *_ctx     = nullptr;
    std::thread    _worker;
    std::atomic<bool> _running{false};

    // use {prompt}
    String _system_prompt = "You are a helpful assistant.";

    // Params cached at load time
    String _model_path;
    int    _n_gpu_layers = 0;
    int    _n_ctx        = 2048;

    void _do_generate(String prompt, int max_tokens);
    String _run_inference(const String &prompt, int max_tokens);

protected:
    static void _bind_methods();

public:
    LlamaNode() = default;
    ~LlamaNode();

    // Call once to load the model (slow — do at scene load)
    bool load_model(String model_path, int n_gpu_layers = 0, int n_ctx = 2048);
    void unload_model();

    // Async — emits `generation_completed(text)` when done
    void generate(String prompt, int max_tokens = 256);

    // Sync version if you want it for simple tests
    String generate_sync(String prompt, int max_tokens = 256);

    // Get/Set _system_prompt
    void set_system_prompt(String prompt) { _system_prompt = prompt; }
    String get_system_prompt() const { return _system_prompt; }

};

} // namespace godot