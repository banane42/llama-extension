#pragma once
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>

struct llama_sampler;

namespace godot {

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerChain
//
// A Resource that holds an ordered Array of LlamaSamplerBase resources.
// Call build() to produce a ready-to-use llama_sampler* chain.
// The caller owns the returned pointer and must call llama_sampler_free().
//
// Typical GDScript usage:
//
//   var chain = LlamaSamplerChain.new()
//   var penalties = LlamaSamplerPenalties.new()
//   penalties.repeat_penalty = 1.1
//   chain.add_sampler(penalties)
//   chain.add_sampler(LlamaSamplerTopK.new())   # uses defaults
//   chain.add_sampler(LlamaSamplerTopP.new())
//   chain.add_sampler(LlamaSamplerTemp.new())
//   chain.add_sampler(LlamaSamplerDist.new())
//   LlamaServer.default_sampler_chain = chain
//
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerChain : public Resource {
    GDCLASS(LlamaSamplerChain, Resource)

private:
    // Ordered list of LlamaSamplerBase resources.
    Array _samplers;

protected:
    static void _bind_methods();

public:
    LlamaSamplerChain()  = default;
    ~LlamaSamplerChain() = default;

    // ── Array management (called from GDScript or C++) ────────────────────────

    /// Append a sampler to the end of the chain.
    void add_sampler(Ref<Resource> sampler);

    /// Insert a sampler at a specific index.
    void insert_sampler(int index, Ref<Resource> sampler);

    /// Remove the sampler at the given index.
    void remove_sampler(int index);

    /// Remove all samplers.
    void clear_samplers();

    /// Return the number of samplers.
    int get_sampler_count() const;

    /// Direct Array property access (enables the inspector Array editor).
    void  set_samplers(const Array &arr) { _samplers = arr; }
    Array get_samplers() const           { return _samplers; }

    // ── Chain construction ────────────────────────────────────────────────────

    /// Build and return a llama_sampler* chain from the current sampler list.
    /// Returns nullptr if the list is empty or contains no valid samplers.
    /// The CALLER is responsible for calling llama_sampler_free() on the result.
    llama_sampler *build() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Registration helper — call once from register_types.cpp.
// ─────────────────────────────────────────────────────────────────────────────
void register_sampler_chain_type();

} // namespace godot
