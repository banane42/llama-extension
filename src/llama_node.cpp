#include "llama_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <string>

// llama.cpp C API
#include "llama.h"
#include "ggml.h"

using namespace std;
using namespace godot;

void LlamaNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "model_path", "n_gpu_layers", "n_ctx"),
                         &LlamaNode::load_model, DEFVAL(0), DEFVAL(2048));
    ClassDB::bind_method(D_METHOD("unload_model"), &LlamaNode::unload_model);
    ClassDB::bind_method(D_METHOD("generate", "prompt", "max_tokens"),
                         &LlamaNode::generate, DEFVAL(256));
    ClassDB::bind_method(D_METHOD("generate_sync", "prompt", "max_tokens"),
                         &LlamaNode::generate_sync, DEFVAL(256));

    ClassDB::bind_method(D_METHOD("set_system_prompt", "prompt"), &LlamaNode::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"), &LlamaNode::get_system_prompt);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt", PROPERTY_HINT_MULTILINE_TEXT), "set_system_prompt", "get_system_prompt");

    ADD_SIGNAL(MethodInfo("generation_completed",
                          PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("generation_failed",
                          PropertyInfo(Variant::STRING, "error")));
}

LlamaNode::~LlamaNode() {
    unload_model();
}

bool LlamaNode::load_model(String model_path, int n_gpu_layers, int n_ctx) {
    unload_model();

    ggml_backend_load_all();          // load all available backends (CUDA, Metal, etc.)
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    _model = llama_model_load_from_file(
        model_path.utf8().get_data(), mparams);

    if (!_model) {
        emit_signal("generation_failed", String("Failed to load model: ") + model_path);
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;

    _ctx = llama_init_from_model(_model, cparams);
    if (!_ctx) {
        llama_model_free(_model);
        _model = nullptr;
        emit_signal("generation_failed", String("Failed to create context"));
        return false;
    }

    _model_path   = model_path;
    _n_gpu_layers = n_gpu_layers;
    _n_ctx        = n_ctx;
    return true;
}

void LlamaNode::unload_model() {
    if (_worker.joinable()) _worker.join();
    if (_ctx)   { llama_free(_ctx);        _ctx   = nullptr; }
    if (_model) { llama_model_free(_model); _model = nullptr; }
}

// ── Core generation logic (shared by sync and async) ──────────────────────────
String LlamaNode::_run_inference(const String &prompt, int max_tokens) {
    if (!LlamaNode::_model || !LlamaNode::_ctx) return "";

    const llama_vocab *vocab = llama_model_get_vocab(LlamaNode::_model);

    std::string prompt_str = prompt.utf8().get_data();
    std::string sys_str = _system_prompt.utf8().get_data();

    std::string p =
        "<|im_start|>system\n"
        + sys_str + "<|im_end|>\n"
        "<|im_start|>user\n"
        + prompt_str +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
    // Tokenize
    int n_prompt = -llama_tokenize(vocab, p.c_str(), p.size(),
                                   nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, p.c_str(), p.size(),
                   tokens.data(), tokens.size(), true, true);

    // Build batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    std::string result;
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));

    for (int i = 0; i < max_tokens; i++) {
        if (llama_decode(LlamaNode::_ctx, batch)) break;

        llama_token new_token = llama_sampler_sample(sampler, LlamaNode::_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256] = {};
        llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        result += buf;

        // Next batch is just the new token
        batch = llama_batch_get_one(&new_token, 1);
    }

    llama_sampler_free(sampler);
	llama_memory_clear(llama_get_memory(LlamaNode::_ctx), true);
    return String(result.c_str());
}

// ── Async ──────────────────────────────────────────────────────────────────────
void LlamaNode::_do_generate(String prompt, int max_tokens) {
    String result = LlamaNode::_run_inference(prompt, max_tokens);
    // Emit signal back on the main thread via call_deferred
    call_deferred("emit_signal", "generation_completed", result);
    _running = false;
}

void LlamaNode::generate(String prompt, int max_tokens) {
    if (!_model) {
        emit_signal("generation_failed", String("Model not loaded"));
        return;
    }
    if (_running) {
        emit_signal("generation_failed", String("Already generating"));
        return;
    }
    _running = true;
    if (_worker.joinable()) _worker.join();
    _worker = std::thread(&LlamaNode::_do_generate, this, prompt, max_tokens);
}

// ── Sync ───────────────────────────────────────────────────────────────────────
String LlamaNode::generate_sync(String prompt, int max_tokens) {
    if (!_model) return "";
    return LlamaNode::_run_inference(prompt, max_tokens);
}