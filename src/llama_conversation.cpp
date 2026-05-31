#include "llama_conversation.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// _bind_methods
// ─────────────────────────────────────────────────────────────────────────────
void LlamaConversation::_bind_methods() {
    // ── Signals ───────────────────────────────────────────────────────────────
    ADD_SIGNAL(MethodInfo("reply_received",    PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("generation_failed", PropertyInfo(Variant::STRING, "error")));

    // ── Generation ────────────────────────────────────────────────────────────
    ClassDB::bind_method(D_METHOD("send",      "user_message", "chain"),
        &LlamaConversation::send,      DEFVAL(Ref<LlamaSamplerChain>()));
    ClassDB::bind_method(D_METHOD("send_sync", "user_message", "chain"),
        &LlamaConversation::send_sync, DEFVAL(Ref<LlamaSamplerChain>()));

    // ── History management ────────────────────────────────────────────────────
    ClassDB::bind_method(D_METHOD("clear"),          &LlamaConversation::clear);
    ClassDB::bind_method(D_METHOD("get_history"),    &LlamaConversation::get_history);
    ClassDB::bind_method(D_METHOD("append_message",  "role", "content"),
        &LlamaConversation::append_message);
    ClassDB::bind_method(D_METHOD("pop_messages",    "count"),
        &LlamaConversation::pop_messages, DEFVAL(1));
    ClassDB::bind_method(D_METHOD("get_message_count"), &LlamaConversation::get_message_count);

    // ── State ─────────────────────────────────────────────────────────────────
    ClassDB::bind_method(D_METHOD("is_generating"), &LlamaConversation::is_generating);

    // ── Config properties ─────────────────────────────────────────────────────
    ClassDB::bind_method(D_METHOD("set_system_prompt", "p"), &LlamaConversation::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"),      &LlamaConversation::get_system_prompt);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt",
        PROPERTY_HINT_MULTILINE_TEXT),
        "set_system_prompt", "get_system_prompt");

    ClassDB::bind_method(D_METHOD("set_max_tokens", "n"), &LlamaConversation::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"),      &LlamaConversation::get_max_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens",
        PROPERTY_HINT_RANGE, "1,65536"),
        "set_max_tokens", "get_max_tokens");

    ClassDB::bind_method(D_METHOD("set_sampler_chain", "chain"), &LlamaConversation::set_sampler_chain);
    ClassDB::bind_method(D_METHOD("get_sampler_chain"),          &LlamaConversation::get_sampler_chain);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sampler_chain",
        PROPERTY_HINT_RESOURCE_TYPE, "LlamaSamplerChain"),
        "set_sampler_chain", "get_sampler_chain");
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
LlamaConversation::~LlamaConversation() {
    if (_worker.joinable()) _worker.join();
    if (_ctx) { llama_free(_ctx); _ctx = nullptr; }
    // _model is borrowed — do NOT free it here.
}

// ─────────────────────────────────────────────────────────────────────────────
// _init  (called by LlamaServer::create_conversation)
// ─────────────────────────────────────────────────────────────────────────────
bool LlamaConversation::_init(llama_model *model, int n_ctx,
                               const String &system_prompt, int max_tokens,
                               const Ref<LlamaSamplerChain> &default_chain) {
    ERR_FAIL_NULL_V_MSG(model, false, "LlamaConversation::_init: model is null.");

    _model         = model;
    _system_prompt = system_prompt;
    _max_tokens    = max_tokens;
    _sampler_chain = default_chain; // may be null — resolved lazily at send() time

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx; // 0 = use model default

    _ctx = llama_init_from_model(_model, cparams);
    if (!_ctx) {
        emit_signal("generation_failed", String("LlamaConversation: failed to create context."));
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt builder
// ─────────────────────────────────────────────────────────────────────────────
// Serialises the complete conversation into a single ChatML string:
//
//   <|im_start|>system
//   {system content}<|im_end|>
//   <|im_start|>user
//   {turn 0}<|im_end|>
//   <|im_start|>assistant
//   {turn 1}<|im_end|>
//   ...
//   <|im_start|>user
//   {user_message}<|im_end|>
//   <|im_start|>assistant
//
// The trailing open assistant tag primes the model to continue.
// ─────────────────────────────────────────────────────────────────────────────
String LlamaConversation::_build_full_prompt(const String &user_message) const {
    String prompt;

    // System turn — extract content from the template if it uses {prompt},
    // otherwise use _system_prompt verbatim as the system block content.
    {
        String sys_content = _system_prompt;
        // If the prompt template still contains {prompt} it was inherited from
        // the fire-and-forget server template; strip the placeholder and any
        // surrounding turn markup so we keep just the system instruction text.
        if (sys_content.contains("{prompt}")) {
            // Remove everything from the first <|im_start|>user occurrence onward.
            int user_pos = sys_content.find("<|im_start|>user");
            if (user_pos != -1) {
                sys_content = sys_content.substr(0, user_pos);
            }
            sys_content = sys_content.replace("{prompt}", "").strip_edges();
        }
        // Unwrap an existing <|im_start|>system ... <|im_end|> wrapper if present
        // so we don't double-wrap.
        if (sys_content.begins_with("<|im_start|>system")) {
            prompt += sys_content;
            if (!sys_content.ends_with("<|im_end|>\n")) prompt += "\n";
        } else {
            prompt += "<|im_start|>system\n" + sys_content + "<|im_end|>\n";
        }
    }

    // Prior turns from history.
    {
        std::lock_guard<std::mutex> lock(_history_mutex);
        for (int i = 0; i < _history.size(); ++i) {
            Dictionary msg = _history[i];
            String role    = msg.get("role",    "user");
            String content = msg.get("content", "");
            prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }

    // Current user turn + open assistant tag to prime generation.
    prompt += "<|im_start|>user\n" + user_message + "<|im_end|>\n<|im_start|>assistant\n";

    return prompt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core inference
// ─────────────────────────────────────────────────────────────────────────────
String LlamaConversation::_run_inference(const String &prompt, llama_sampler *sampler) {
    if (!_model || !_ctx) return "";

    const llama_vocab *vocab = llama_model_get_vocab(_model);

    std::string p = prompt.utf8().get_data();

    // Tokenise
    int n_prompt = -llama_tokenize(vocab, p.c_str(), p.size(),
                                   nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, p.c_str(), p.size(),
                   tokens.data(), tokens.size(), true, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    std::string result;
    for (int i = 0; i < _max_tokens; ++i) {
        if (llama_decode(_ctx, batch)) break;

        llama_token new_token = llama_sampler_sample(sampler, _ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256] = {};
        llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        result += buf;

        batch = llama_batch_get_one(&new_token, 1);
    }

    // Clear KV cache so the next turn starts fresh (no partial-sequence reuse yet).
    llama_memory_clear(llama_get_memory(_ctx), true);

    return String(result.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampler resolution — priority: conversation chain > call-site chain > error
// ─────────────────────────────────────────────────────────────────────────────
// (Intentionally a free helper inside this TU so it stays private to the file.)
static llama_sampler *resolve_chain(LlamaConversation *conv,
                                    const Ref<LlamaSamplerChain> &call_chain,
                                    const Ref<LlamaSamplerChain> &conv_chain) {
    // 1. Conversation-level chain takes highest priority.
    if (conv_chain.is_valid()) {
        llama_sampler *s = conv_chain->build();
        if (s) return s;
    }
    // 2. Chain supplied at the call site (may be the server's default, forwarded
    //    through create_conversation → _init → _sampler_chain, or explicitly
    //    provided by GDScript at each send() call).
    if (call_chain.is_valid()) {
        llama_sampler *s = call_chain->build();
        if (s) return s;
    }
    // 3. Nothing usable.
    conv->emit_signal("generation_failed",
        String("LlamaConversation: no sampler chain available. "
               "Set sampler_chain on the conversation or pass one to send()."));
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Async send
// ─────────────────────────────────────────────────────────────────────────────
void LlamaConversation::_do_send(String user_message, Ref<LlamaSamplerChain> chain) {
    llama_sampler *sampler = resolve_chain(this, chain, _sampler_chain);
    if (!sampler) {
        _running = false;
        return;
    }

    String full_prompt = _build_full_prompt(user_message);
    String reply       = _run_inference(full_prompt, sampler);
    llama_sampler_free(sampler);

    if (reply.is_empty()) {
        // _run_inference already handles nullptr model/ctx; emit a generic error
        // only if we actually got nothing back (not just an early EOS).
        // An empty reply after a valid decode is unusual but not fatal — treat
        // it as a successful (but empty) reply so history stays consistent.
    }

    // Commit both turns to history under the lock.
    {
        std::lock_guard<std::mutex> lock(_history_mutex);
        Dictionary user_turn;
        user_turn["role"]    = String("user");
        user_turn["content"] = user_message;
        _history.push_back(user_turn);

        Dictionary asst_turn;
        asst_turn["role"]    = String("assistant");
        asst_turn["content"] = reply;
        _history.push_back(asst_turn);
    }

    call_deferred("emit_signal", "reply_received", reply);
    _running = false;
}

void LlamaConversation::send(String user_message, Ref<LlamaSamplerChain> chain) {
    ERR_FAIL_NULL_MSG(_ctx, "LlamaConversation: context not initialised. "
                            "Use LlamaServer.create_conversation() to create conversations.");
    if (_running) {
        emit_signal("generation_failed",
            String("LlamaConversation: already generating. "
                   "Wait for reply_received before calling send() again."));
        return;
    }

    _running = true;
    if (_worker.joinable()) _worker.join();
    _worker = std::thread(&LlamaConversation::_do_send, this, user_message, chain);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sync send
// ─────────────────────────────────────────────────────────────────────────────
String LlamaConversation::send_sync(String user_message, Ref<LlamaSamplerChain> chain) {
    ERR_FAIL_NULL_V_MSG(_ctx, "",
        "LlamaConversation: context not initialised. "
        "Use LlamaServer.create_conversation() to create conversations.");

    llama_sampler *sampler = resolve_chain(this, chain, _sampler_chain);
    if (!sampler) return "";

    String full_prompt = _build_full_prompt(user_message);
    String reply       = _run_inference(full_prompt, sampler);
    llama_sampler_free(sampler);

    {
        std::lock_guard<std::mutex> lock(_history_mutex);
        Dictionary user_turn;
        user_turn["role"]    = String("user");
        user_turn["content"] = user_message;
        _history.push_back(user_turn);

        Dictionary asst_turn;
        asst_turn["role"]    = String("assistant");
        asst_turn["content"] = reply;
        _history.push_back(asst_turn);
    }

    return reply;
}

// ─────────────────────────────────────────────────────────────────────────────
// History management
// ─────────────────────────────────────────────────────────────────────────────
void LlamaConversation::clear() {
    if (_worker.joinable()) _worker.join();
    {
        std::lock_guard<std::mutex> lock(_history_mutex);
        _history.clear();
    }
    if (_ctx) llama_memory_clear(llama_get_memory(_ctx), true);
}

Array LlamaConversation::get_history() const {
    std::lock_guard<std::mutex> lock(_history_mutex);
    return _history.duplicate(); // return a copy so callers can't mutate internal state
}

void LlamaConversation::append_message(String role, String content) {
    Dictionary msg;
    msg["role"]    = role;
    msg["content"] = content;
    std::lock_guard<std::mutex> lock(_history_mutex);
    _history.push_back(msg);
}

void LlamaConversation::pop_messages(int count) {
    ERR_FAIL_COND_MSG(count < 1, "LlamaConversation::pop_messages: count must be >= 1.");
    std::lock_guard<std::mutex> lock(_history_mutex);
    for (int i = 0; i < count && !_history.is_empty(); ++i) {
        _history.resize(_history.size() - 1);
    }
}

int LlamaConversation::get_message_count() const {
    std::lock_guard<std::mutex> lock(_history_mutex);
    return _history.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// Config setters
// ─────────────────────────────────────────────────────────────────────────────

// Changing the system prompt mid-conversation takes effect on the next turn.
// The history is not replayed — previous turns remain in the log unchanged.
void LlamaConversation::set_system_prompt(String p) {
    _system_prompt = p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────
void godot::register_conversation_type() {
    ClassDB::register_class<LlamaConversation>();
}
