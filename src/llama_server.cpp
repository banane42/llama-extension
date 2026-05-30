#include "llama_server.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <string>

#include "llama.h"
#include "ggml.h"

using namespace godot;

// ── Singleton ──────────────────────────────────────────────────────────────────
LlamaServer *LlamaServer::_singleton = nullptr;

LlamaServer *LlamaServer::get_singleton() {
    return _singleton;
}

LlamaServer::LlamaServer() {
    _singleton = this;
}

LlamaServer::~LlamaServer() {
    unload_model();
    _singleton = nullptr;
}

// ── Bind methods ───────────────────────────────────────────────────────────────
void LlamaServer::_bind_methods() {

    // Model lifecycle
    ClassDB::bind_method(D_METHOD("load_model", "model_path", "n_gpu_layers", "n_ctx"), 
        &LlamaServer::load_model, DEFVAL(-1), DEFVAL(0));
    ClassDB::bind_method(D_METHOD("unload_model"),             &LlamaServer::unload_model);
    ClassDB::bind_method(D_METHOD("is_model_loaded"),          &LlamaServer::is_model_loaded);

    // Generation
    ClassDB::bind_method(D_METHOD("generate", "prompt"),       &LlamaServer::generate);
    ClassDB::bind_method(D_METHOD("generate_sync", "prompt"),  &LlamaServer::generate_sync);

    // Signals
    ADD_SIGNAL(MethodInfo("generation_completed", PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("generation_failed",    PropertyInfo(Variant::STRING, "error")));

    // ── Properties ─────────────────────────────────────────────────────────────

    // Model config
    ClassDB::bind_method(D_METHOD("set_model_path", "path"),   &LlamaServer::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"),           &LlamaServer::get_model_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"),
                 "set_model_path", "get_model_path");

    ClassDB::bind_method(D_METHOD("set_max_tokens", "n"),      &LlamaServer::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"),           &LlamaServer::get_max_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens"),
                 "set_max_tokens", "get_max_tokens");

    // Sampler config
    ClassDB::bind_method(D_METHOD("set_top_k", "n"),           &LlamaServer::set_top_k);
    ClassDB::bind_method(D_METHOD("get_top_k"),                &LlamaServer::get_top_k);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k"),
                 "set_top_k", "get_top_k");

    ClassDB::bind_method(D_METHOD("set_top_p", "p"),           &LlamaServer::set_top_p);
    ClassDB::bind_method(D_METHOD("get_top_p"),                &LlamaServer::get_top_p);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p"),
                 "set_top_p", "get_top_p");

    ClassDB::bind_method(D_METHOD("set_temperature", "t"),     &LlamaServer::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"),          &LlamaServer::get_temperature);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature"),
                 "set_temperature", "get_temperature");

    // Penalty config
    ClassDB::bind_method(D_METHOD("set_penalty_last_n", "n"),  &LlamaServer::set_penalty_last_n);
    ClassDB::bind_method(D_METHOD("get_penalty_last_n"),       &LlamaServer::get_penalty_last_n);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "penalty_last_n"),
                 "set_penalty_last_n", "get_penalty_last_n");

    ClassDB::bind_method(D_METHOD("set_penalty_repeat", "p"),  &LlamaServer::set_penalty_repeat);
    ClassDB::bind_method(D_METHOD("get_penalty_repeat"),       &LlamaServer::get_penalty_repeat);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "penalty_repeat"),
                 "set_penalty_repeat", "get_penalty_repeat");

    // System prompt
    ClassDB::bind_method(D_METHOD("set_system_prompt", "p"),   &LlamaServer::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"),        &LlamaServer::get_system_prompt);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt", PROPERTY_HINT_MULTILINE_TEXT),
                 "set_system_prompt", "get_system_prompt");
}

bool LlamaServer::load_model(String model_path, int n_gpu_layers, int n_ctx) {
    unload_model();

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    _model = llama_model_load_from_file(model_path.utf8().get_data(), mparams);
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

    _model_path = model_path;
    return true;
}

void LlamaServer::unload_model() {
    if (_worker.joinable()) _worker.join();
    if (_ctx)   { llama_free(_ctx);         _ctx   = nullptr; }
    if (_model) { llama_model_free(_model); _model = nullptr; }
}

// ── Prompt building ────────────────────────────────────────────────────────────
// Replaces {prompt} in the system prompt template with the ChatML user turn.
// The template is model-agnostic — just swap out the system prompt string
// to match whatever chat format your GGUF expects.
String LlamaServer::_build_prompt(const String &user_prompt) const {
    String user_turn =
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";

    String result = _system_prompt;
    result = result.replace("{prompt}", user_turn);
    return result;
}

// ── Core inference ─────────────────────────────────────────────────────────────
String LlamaServer::_run_inference(const String &prompt) {
    if (!_model || !_ctx) return "";

    const llama_vocab *vocab = llama_model_get_vocab(_model);

    std::string p = _build_prompt(prompt).utf8().get_data();

    // Tokenize
    int n_prompt = -llama_tokenize(vocab, p.c_str(), p.size(),
                                   nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, p.c_str(), p.size(),
                   tokens.data(), tokens.size(), true, true);

    // Build initial batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // Build sampler chain
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(_penalty_last_n, _penalty_repeat, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(_top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(_top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(_temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string result;

    for (int i = 0; i < _max_tokens; i++) {
        if (llama_decode(_ctx, batch)) break;

        llama_token new_token = llama_sampler_sample(sampler, _ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256] = {};
        llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        result += buf;

        batch = llama_batch_get_one(&new_token, 1);
    }

    llama_sampler_free(sampler);
    llama_memory_clear(llama_get_memory(_ctx), true);

    return String(result.c_str());
}

// ── Async generation ───────────────────────────────────────────────────────────
void LlamaServer::_do_generate(String prompt) {
    String result = _run_inference(prompt);
    call_deferred("emit_signal", "generation_completed", result);
    _running = false;
}

void LlamaServer::generate(String prompt) {
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
    _worker = std::thread(&LlamaServer::_do_generate, this, prompt);
}

// ── Sync generation ────────────────────────────────────────────────────────────
String LlamaServer::generate_sync(String prompt) {
    if (!_model) return "";
    return _run_inference(prompt);
}