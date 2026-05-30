#include "llama_sampler_chain.h"
#include "llama_samplers.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "llama.h"

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// _bind_methods
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerChain::_bind_methods() {
    // Array property — shows up in the inspector as a typed array.
    ClassDB::bind_method(D_METHOD("set_samplers", "samplers"), &LlamaSamplerChain::set_samplers);
    ClassDB::bind_method(D_METHOD("get_samplers"),             &LlamaSamplerChain::get_samplers);
    ADD_PROPERTY(
        PropertyInfo(Variant::ARRAY, "samplers",
            PROPERTY_HINT_ARRAY_TYPE, "LlamaSamplerBase"),
        "set_samplers", "get_samplers");

    // Convenience mutators accessible from GDScript.
    ClassDB::bind_method(D_METHOD("add_sampler",    "sampler"),        &LlamaSamplerChain::add_sampler);
    ClassDB::bind_method(D_METHOD("insert_sampler", "index","sampler"),&LlamaSamplerChain::insert_sampler);
    ClassDB::bind_method(D_METHOD("remove_sampler", "index"),          &LlamaSamplerChain::remove_sampler);
    ClassDB::bind_method(D_METHOD("clear_samplers"),                   &LlamaSamplerChain::clear_samplers);
    ClassDB::bind_method(D_METHOD("get_sampler_count"),                &LlamaSamplerChain::get_sampler_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Array management
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerChain::add_sampler(Ref<Resource> sampler) {
    ERR_FAIL_COND_MSG(sampler.is_null(), "LlamaSamplerChain: cannot add a null sampler.");
    ERR_FAIL_COND_MSG(!sampler->is_class("LlamaSamplerBase"),
        "LlamaSamplerChain: resource is not a LlamaSamplerBase subclass.");
    _samplers.push_back(sampler);
}

void LlamaSamplerChain::insert_sampler(int index, Ref<Resource> sampler) {
    ERR_FAIL_COND_MSG(sampler.is_null(), "LlamaSamplerChain: cannot insert a null sampler.");
    ERR_FAIL_COND_MSG(!sampler->is_class("LlamaSamplerBase"),
        "LlamaSamplerChain: resource is not a LlamaSamplerBase subclass.");
    ERR_FAIL_INDEX(index, _samplers.size() + 1);
    _samplers.insert(index, sampler);
}

void LlamaSamplerChain::remove_sampler(int index) {
    ERR_FAIL_INDEX(index, _samplers.size());
    _samplers.remove_at(index);
}

void LlamaSamplerChain::clear_samplers() {
    _samplers.clear();
}

int LlamaSamplerChain::get_sampler_count() const {
    return _samplers.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// build() — assemble the llama.cpp sampler chain
// ─────────────────────────────────────────────────────────────────────────────
llama_sampler *LlamaSamplerChain::build() const {
    if (_samplers.is_empty()) {
        UtilityFunctions::push_warning("LlamaSamplerChain::build(): sampler list is empty.");
        return nullptr;
    }

    llama_sampler *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    int added = 0;
    for (int i = 0; i < _samplers.size(); ++i) {
        Ref<LlamaSamplerBase> s = _samplers[i];
        if (s.is_null()) {
            UtilityFunctions::push_warning(
                vformat("LlamaSamplerChain::build(): null entry at index %d, skipping.", i));
            continue;
        }
        s->add_to_chain(chain);
        ++added;
    }

    if (added == 0) {
        llama_sampler_free(chain);
        UtilityFunctions::push_warning("LlamaSamplerChain::build(): no valid samplers were added.");
        return nullptr;
    }

    return chain;
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────
void godot::register_sampler_chain_type() {
    ClassDB::register_class<LlamaSamplerChain>();
}
