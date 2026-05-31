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

    // Conversation factory
    ClassDB::bind_method(D_METHOD("create_conversation", "n_ctx"),
        &LlamaServer::create_conversation, DEFVAL(0));

    // Model lifecycle
    ClassDB::bind_method(D_METHOD("load_model", "model_path", "n_gpu_layers", "n_ctx"),
        &LlamaServer::load_model, DEFVAL(-1), DEFVAL(0));
    ClassDB::bind_method(D_METHOD("unload_model"),    &LlamaServer::unload_model);
    ClassDB::bind_method(D_METHOD("is_model_loaded"), &LlamaServer::is_model_loaded);

    // Generation — chain argument is optional (null = use server default)
    ClassDB::bind_method(D_METHOD("generate",      "prompt", "chain"), &LlamaServer::generate,      DEFVAL(Ref<LlamaSamplerChain>()));
    ClassDB::bind_method(D_METHOD("generate_sync", "prompt", "chain"), &LlamaServer::generate_sync, DEFVAL(Ref<LlamaSamplerChain>()));

    // Signals
    ADD_SIGNAL(MethodInfo("generation_completed", PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("generation_failed",    PropertyInfo(Variant::STRING, "error")));

    // ── Properties ─────────────────────────────────────────────────────────────

    // Model config
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &LlamaServer::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"),         &LlamaServer::get_model_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path",
        PROPERTY_HINT_FILE, "*.gguf"),
        "set_model_path", "get_model_path");

    ClassDB::bind_method(D_METHOD("set_max_tokens", "n"), &LlamaServer::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"),      &LlamaServer::get_max_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens",
        PROPERTY_HINT_RANGE, "1,65536"),
        "set_max_tokens", "get_max_tokens");

    // Default sampler chain — must be assigned before generation
    ClassDB::bind_method(D_METHOD("set_default_sampler_chain", "chain"), &LlamaServer::set_default_sampler_chain);
    ClassDB::bind_method(D_METHOD("get_default_sampler_chain"),          &LlamaServer::get_default_sampler_chain);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "default_sampler_chain",
        PROPERTY_HINT_RESOURCE_TYPE, "LlamaSamplerChain"),
        "set_default_sampler_chain", "get_default_sampler_chain");

    // System prompt
    ClassDB::bind_method(D_METHOD("set_system_prompt", "p"), &LlamaServer::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"),      &LlamaServer::get_system_prompt);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt",
        PROPERTY_HINT_MULTILINE_TEXT),
        "set_system_prompt", "get_system_prompt");
}

// ── Model lifecycle ────────────────────────────────────────────────────────────
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
String LlamaServer::_build_prompt(const String &user_prompt) const {
    String user_turn =
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";

    String result = _system_prompt;
    result = result.replace("{prompt}", user_turn);
    return result;
}

// ── Sampler chain resolution ───────────────────────────────────────────────────
// Priority: per-call override → server default → error.
// Returns a freshly-built llama_sampler* the caller must free, or nullptr on failure.
llama_sampler *LlamaServer::_resolve_sampler_chain(const Ref<LlamaSamplerChain> &override_chain) {
    // 1. Per-call override takes priority.
    if (override_chain.is_valid()) {
        llama_sampler *built = override_chain->build();
        if (built) return built;
        // build() already pushed a warning — treat as if no override was given
        // and continue to the server default rather than silently succeeding.
    }

    // 2. Server default.
    if (_default_sampler_chain.is_valid()) {
        llama_sampler *built = _default_sampler_chain->build();
        if (built) return built;
    }

    // 3. Nothing usable — hard error.
    const String msg = _default_sampler_chain.is_null()
        ? "No sampler chain set. Assign a LlamaSamplerChain to default_sampler_chain "
          "or pass one directly to generate() / generate_sync()."
        : "default_sampler_chain is set but build() returned no valid samplers. "
          "Check that the chain contains at least one sampler.";

    emit_signal("generation_failed", msg);
    return nullptr;
}

// ── Core inference ─────────────────────────────────────────────────────────────
String LlamaServer::_run_inference(const String &prompt, llama_sampler *sampler) {
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

    llama_memory_clear(llama_get_memory(_ctx), true);

    return String(result.c_str());
}

// ── Async generation ───────────────────────────────────────────────────────────
// The chain is resolved on the calling thread (before the worker spawns) so that
// any resolution error emits generation_failed on the main thread, not the worker.
void LlamaServer::_do_generate(String prompt, Ref<LlamaSamplerChain> chain) {
    // chain was already validated by generate() — build the llama_sampler* here
    // so its lifetime is wholly contained within the worker thread.
    llama_sampler *sampler = _resolve_sampler_chain(chain);
    if (!sampler) {
        _running = false;
        return;
    }

    String result = _run_inference(prompt, sampler);
    llama_sampler_free(sampler);
    call_deferred("emit_signal", "generation_completed", result);
    _running = false;
}

void LlamaServer::generate(String prompt, Ref<LlamaSamplerChain> chain) {
    if (!_model) {
        emit_signal("generation_failed", String("Model not loaded"));
        return;
    }
    if (_running) {
        emit_signal("generation_failed", String("Already generating"));
        return;
    }

    // Validate the chain on the calling thread before we commit to the worker.
    // We pass the Ref<> into the thread rather than the raw pointer so the
    // resource stays alive for the duration of the worker's execution.
    const Ref<LlamaSamplerChain> &resolved = chain.is_valid() ? chain : _default_sampler_chain;
    if (resolved.is_null()) {
        emit_signal("generation_failed",
            String("No sampler chain set. Assign a LlamaSamplerChain to default_sampler_chain "
                   "or pass one directly to generate()."));
        return;
    }

    _running = true;
    if (_worker.joinable()) _worker.join();
    _worker = std::thread(&LlamaServer::_do_generate, this, prompt, resolved);
}

// ── Conversation factory ───────────────────────────────────────────────────────
Ref<LlamaConversation> LlamaServer::create_conversation(int n_ctx) {
    if (!_model) {
        emit_signal("generation_failed",
            String("LlamaServer::create_conversation: model not loaded."));
        return Ref<LlamaConversation>();
    }

    Ref<LlamaConversation> conv;
    conv.instantiate();

    if (!conv->_init(_model, n_ctx, _system_prompt, _max_tokens, _default_sampler_chain)) {
        // _init already emitted generation_failed on the conversation object;
        // emit it on the server too so callers that only listen to the server signal
        // are also notified.
        emit_signal("generation_failed",
            String("LlamaServer::create_conversation: context initialisation failed."));
        return Ref<LlamaConversation>();
    }

    return conv;
}

// ── Sync generation ────────────────────────────────────────────────────────────
String LlamaServer::generate_sync(String prompt, Ref<LlamaSamplerChain> chain) {
    if (!_model) {
        emit_signal("generation_failed", String("Model not loaded"));
        return "";
    }

    llama_sampler *sampler = _resolve_sampler_chain(chain);
    if (!sampler) return "";

    String result = _run_inference(prompt, sampler);
    llama_sampler_free(sampler);
    return result;
}
