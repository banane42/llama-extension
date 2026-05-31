#pragma once
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <thread>
#include <atomic>
#include <mutex>

#include "llama_sampler_chain.h"

struct llama_model;
struct llama_context;

namespace godot {

// ─────────────────────────────────────────────────────────────────────────────
// LlamaConversation
//
// A persistent, stateful conversation thread backed by its own llama_context.
// Each turn appends to an internal message history that is re-serialised into
// a full ChatML prompt on every call, giving the model complete context.
//
// Create one via LlamaServer::create_conversation(); do not instantiate directly.
//
// Typical GDScript usage:
//
//   var conv = LlamaServer.create_conversation()
//   conv.reply_received.connect(func(text): print("Bot: ", text))
//   conv.send("Hello! Who are you?")
//   # later...
//   conv.send("What did I just ask you?")   # model sees full history
//   conv.clear()                            # wipe history, keep context alive
//
// ─────────────────────────────────────────────────────────────────────────────
class LlamaConversation : public RefCounted {
    GDCLASS(LlamaConversation, RefCounted)

private:
    // ── llama.cpp state (not owned — model belongs to LlamaServer) ────────────
    llama_model   *_model = nullptr;   // borrowed from LlamaServer; never freed here
    llama_context *_ctx   = nullptr;   // owned; freed in destructor

    // ── Worker thread ─────────────────────────────────────────────────────────
    std::thread       _worker;
    std::atomic<bool> _running{false};

    // ── History ───────────────────────────────────────────────────────────────
    // Each element is a Dictionary with keys "role" (String) and "content" (String).
    Array _history;
    mutable std::mutex _history_mutex;

    // ── Config (copied from server at construction time) ──────────────────────
    String _system_prompt;
    int    _max_tokens = 256;

    // Per-conversation sampler chain override. Falls back to the server default
    // if null (resolved at send() time, not stored as a raw pointer).
    Ref<LlamaSamplerChain> _sampler_chain;

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Serialise the full conversation (system prompt + all history + pending
    /// user turn) into a single ChatML string ready for tokenisation.
    String _build_full_prompt(const String &user_message) const;

    /// Core inference: tokenise prompt, run decode loop, return generated text.
    /// `sampler` must be non-null; caller is responsible for freeing it.
    String _run_inference(const String &prompt, llama_sampler *sampler);

    /// Worker body for async send().
    void _do_send(String user_message, Ref<LlamaSamplerChain> chain);

protected:
    static void _bind_methods();

public:
    LlamaConversation()  = default;
    ~LlamaConversation() override;

    // ── Lifecycle (called by LlamaServer::create_conversation) ────────────────

    /// Initialise with a borrowed model pointer and a fresh context.
    /// Returns false and emits generation_failed if context creation fails.
    bool _init(llama_model *model, int n_ctx,
               const String &system_prompt, int max_tokens,
               const Ref<LlamaSamplerChain> &default_chain);

    // ── Generation ────────────────────────────────────────────────────────────

    /// Async: appends the user turn, generates a reply, emits reply_received.
    /// Uses this conversation's sampler chain if set, otherwise falls back to
    /// the chain passed here (which the server provides from its own default).
    void   send(String user_message,
                Ref<LlamaSamplerChain> chain = Ref<LlamaSamplerChain>());

    /// Blocking variant. Returns the assistant reply, or "" on error.
    String send_sync(String user_message,
                     Ref<LlamaSamplerChain> chain = Ref<LlamaSamplerChain>());

    // ── History management ────────────────────────────────────────────────────

    /// Erase all turns. The context KV cache is also cleared.
    /// Does not affect the system prompt, config, or sampler chain.
    void clear();

    /// Return a copy of the history as an Array of Dictionaries:
    ///   [ { "role": "user", "content": "Hello" },
    ///     { "role": "assistant", "content": "Hi!" }, ... ]
    Array get_history() const;

    /// Inject an arbitrary turn directly into the history without inference.
    /// Useful for loading a prior conversation from disk.
    void append_message(String role, String content);

    /// Remove the last N turns (default 1). Useful for retry / regenerate.
    void pop_messages(int count = 1);

    /// Return the number of turns in the history.
    int get_message_count() const;

    // ── Per-conversation config ───────────────────────────────────────────────

    void   set_system_prompt(String p);
    String get_system_prompt() const  { return _system_prompt; }

    void set_max_tokens(int n)        { _max_tokens = n; }
    int  get_max_tokens() const       { return _max_tokens; }

    /// Optional sampler chain specific to this conversation.
    /// When set it takes priority over any chain passed to send() / send_sync()
    /// and over the server-level default.
    void                   set_sampler_chain(Ref<LlamaSamplerChain> chain) { _sampler_chain = chain; }
    Ref<LlamaSamplerChain> get_sampler_chain() const                       { return _sampler_chain; }

    /// True while an async send() is in progress.
    bool is_generating() const { return _running.load(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Registration helper — call once from register_types.cpp.
// ─────────────────────────────────────────────────────────────────────────────
void register_conversation_type();

} // namespace godot
