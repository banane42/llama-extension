#include "llama_samplers.h"
#include <godot_cpp/core/class_db.hpp>

#include "llama.h"

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static uint32_t resolve_seed(int seed) {
    return (seed < 0) ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerGreedy
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerGreedy::_bind_methods() {
    // No parameters — nothing to bind.
}

void LlamaSamplerGreedy::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerDist
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerDist::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_seed", "seed"), &LlamaSamplerDist::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"),         &LlamaSamplerDist::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed",
        PROPERTY_HINT_RANGE, "-1,2147483647"),
        "set_seed", "get_seed");
}

void LlamaSamplerDist::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_dist(resolve_seed(_seed)));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerTemp
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerTemp::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_temperature", "t"), &LlamaSamplerTemp::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"),      &LlamaSamplerTemp::get_temperature);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature",
        PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
        "set_temperature", "get_temperature");
}

void LlamaSamplerTemp::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(_temperature));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerTempExt
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerTempExt::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_temperature", "t"), &LlamaSamplerTempExt::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"),      &LlamaSamplerTempExt::get_temperature);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature",
        PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
        "set_temperature", "get_temperature");

    ClassDB::bind_method(D_METHOD("set_delta", "d"), &LlamaSamplerTempExt::set_delta);
    ClassDB::bind_method(D_METHOD("get_delta"),      &LlamaSamplerTempExt::get_delta);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "delta",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_delta", "get_delta");

    ClassDB::bind_method(D_METHOD("set_exponent", "e"), &LlamaSamplerTempExt::set_exponent);
    ClassDB::bind_method(D_METHOD("get_exponent"),      &LlamaSamplerTempExt::get_exponent);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "exponent",
        PROPERTY_HINT_RANGE, "0.0,4.0,0.01"),
        "set_exponent", "get_exponent");
}

void LlamaSamplerTempExt::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain,
        llama_sampler_init_temp_ext(_temperature, _delta, _exponent));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerTopK
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerTopK::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_top_k", "k"), &LlamaSamplerTopK::set_top_k);
    ClassDB::bind_method(D_METHOD("get_top_k"),      &LlamaSamplerTopK::get_top_k);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k",
        PROPERTY_HINT_RANGE, "0,1000"),
        "set_top_k", "get_top_k");
}

void LlamaSamplerTopK::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(_top_k));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerTopP
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerTopP::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_top_p", "p"),       &LlamaSamplerTopP::set_top_p);
    ClassDB::bind_method(D_METHOD("get_top_p"),            &LlamaSamplerTopP::get_top_p);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_top_p", "get_top_p");

    ClassDB::bind_method(D_METHOD("set_min_keep", "n"),    &LlamaSamplerTopP::set_min_keep);
    ClassDB::bind_method(D_METHOD("get_min_keep"),         &LlamaSamplerTopP::get_min_keep);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_keep",
        PROPERTY_HINT_RANGE, "0,100"),
        "set_min_keep", "get_min_keep");
}

void LlamaSamplerTopP::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(_top_p, _min_keep));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerMinP
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerMinP::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_min_p", "p"),       &LlamaSamplerMinP::set_min_p);
    ClassDB::bind_method(D_METHOD("get_min_p"),            &LlamaSamplerMinP::get_min_p);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_p",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_min_p", "get_min_p");

    ClassDB::bind_method(D_METHOD("set_min_keep", "n"),    &LlamaSamplerMinP::set_min_keep);
    ClassDB::bind_method(D_METHOD("get_min_keep"),         &LlamaSamplerMinP::get_min_keep);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_keep",
        PROPERTY_HINT_RANGE, "0,100"),
        "set_min_keep", "get_min_keep");
}

void LlamaSamplerMinP::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(_min_p, _min_keep));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerTypical
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerTypical::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_p", "p"),           &LlamaSamplerTypical::set_p);
    ClassDB::bind_method(D_METHOD("get_p"),                &LlamaSamplerTypical::get_p);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "p",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_p", "get_p");

    ClassDB::bind_method(D_METHOD("set_min_keep", "n"),    &LlamaSamplerTypical::set_min_keep);
    ClassDB::bind_method(D_METHOD("get_min_keep"),         &LlamaSamplerTypical::get_min_keep);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_keep",
        PROPERTY_HINT_RANGE, "0,100"),
        "set_min_keep", "get_min_keep");
}

void LlamaSamplerTypical::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain, llama_sampler_init_typical(_p, _min_keep));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerPenalties
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerPenalties::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_last_n", "n"),           &LlamaSamplerPenalties::set_last_n);
    ClassDB::bind_method(D_METHOD("get_last_n"),                &LlamaSamplerPenalties::get_last_n);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "last_n",
        PROPERTY_HINT_RANGE, "-1,2048"),
        "set_last_n", "get_last_n");

    ClassDB::bind_method(D_METHOD("set_repeat_penalty", "p"),   &LlamaSamplerPenalties::set_repeat_penalty);
    ClassDB::bind_method(D_METHOD("get_repeat_penalty"),        &LlamaSamplerPenalties::get_repeat_penalty);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "repeat_penalty",
        PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
        "set_repeat_penalty", "get_repeat_penalty");

    ClassDB::bind_method(D_METHOD("set_freq_penalty", "p"),     &LlamaSamplerPenalties::set_freq_penalty);
    ClassDB::bind_method(D_METHOD("get_freq_penalty"),          &LlamaSamplerPenalties::get_freq_penalty);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "freq_penalty",
        PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
        "set_freq_penalty", "get_freq_penalty");

    ClassDB::bind_method(D_METHOD("set_present_penalty", "p"),  &LlamaSamplerPenalties::set_present_penalty);
    ClassDB::bind_method(D_METHOD("get_present_penalty"),       &LlamaSamplerPenalties::get_present_penalty);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "present_penalty",
        PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
        "set_present_penalty", "get_present_penalty");
}

void LlamaSamplerPenalties::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(_last_n, _repeat_penalty, _freq_penalty, _present_penalty));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerMirostat
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerMirostat::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_n_vocab", "n"),  &LlamaSamplerMirostat::set_n_vocab);
    ClassDB::bind_method(D_METHOD("get_n_vocab"),       &LlamaSamplerMirostat::get_n_vocab);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_vocab",
        PROPERTY_HINT_RANGE, "0,1000000"),
        "set_n_vocab", "get_n_vocab");

    ClassDB::bind_method(D_METHOD("set_seed", "seed"),  &LlamaSamplerMirostat::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"),          &LlamaSamplerMirostat::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed",
        PROPERTY_HINT_RANGE, "-1,2147483647"),
        "set_seed", "get_seed");

    ClassDB::bind_method(D_METHOD("set_tau", "tau"),    &LlamaSamplerMirostat::set_tau);
    ClassDB::bind_method(D_METHOD("get_tau"),           &LlamaSamplerMirostat::get_tau);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tau",
        PROPERTY_HINT_RANGE, "0.0,20.0,0.1"),
        "set_tau", "get_tau");

    ClassDB::bind_method(D_METHOD("set_eta", "eta"),    &LlamaSamplerMirostat::set_eta);
    ClassDB::bind_method(D_METHOD("get_eta"),           &LlamaSamplerMirostat::get_eta);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "eta",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_eta", "get_eta");

    ClassDB::bind_method(D_METHOD("set_m", "m"),        &LlamaSamplerMirostat::set_m);
    ClassDB::bind_method(D_METHOD("get_m"),             &LlamaSamplerMirostat::get_m);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "m",
        PROPERTY_HINT_RANGE, "1,1000"),
        "set_m", "get_m");
}

void LlamaSamplerMirostat::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain,
        llama_sampler_init_mirostat(_n_vocab, resolve_seed(_seed), _tau, _eta, _m));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerMirostatV2
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerMirostatV2::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_seed", "seed"),  &LlamaSamplerMirostatV2::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"),          &LlamaSamplerMirostatV2::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed",
        PROPERTY_HINT_RANGE, "-1,2147483647"),
        "set_seed", "get_seed");

    ClassDB::bind_method(D_METHOD("set_tau", "tau"),    &LlamaSamplerMirostatV2::set_tau);
    ClassDB::bind_method(D_METHOD("get_tau"),           &LlamaSamplerMirostatV2::get_tau);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tau",
        PROPERTY_HINT_RANGE, "0.0,20.0,0.1"),
        "set_tau", "get_tau");

    ClassDB::bind_method(D_METHOD("set_eta", "eta"),    &LlamaSamplerMirostatV2::set_eta);
    ClassDB::bind_method(D_METHOD("get_eta"),           &LlamaSamplerMirostatV2::get_eta);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "eta",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_eta", "get_eta");
}

void LlamaSamplerMirostatV2::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain,
        llama_sampler_init_mirostat_v2(resolve_seed(_seed), _tau, _eta));
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaSamplerXTC
// ─────────────────────────────────────────────────────────────────────────────
void LlamaSamplerXTC::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_probability", "p"), &LlamaSamplerXTC::set_probability);
    ClassDB::bind_method(D_METHOD("get_probability"),      &LlamaSamplerXTC::get_probability);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "probability",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_probability", "get_probability");

    ClassDB::bind_method(D_METHOD("set_threshold", "t"),   &LlamaSamplerXTC::set_threshold);
    ClassDB::bind_method(D_METHOD("get_threshold"),        &LlamaSamplerXTC::get_threshold);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "threshold",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
        "set_threshold", "get_threshold");

    ClassDB::bind_method(D_METHOD("set_min_keep", "n"),    &LlamaSamplerXTC::set_min_keep);
    ClassDB::bind_method(D_METHOD("get_min_keep"),         &LlamaSamplerXTC::get_min_keep);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_keep",
        PROPERTY_HINT_RANGE, "0,100"),
        "set_min_keep", "get_min_keep");

    ClassDB::bind_method(D_METHOD("set_seed", "seed"),     &LlamaSamplerXTC::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"),             &LlamaSamplerXTC::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed",
        PROPERTY_HINT_RANGE, "-1,2147483647"),
        "set_seed", "get_seed");
}

void LlamaSamplerXTC::add_to_chain(llama_sampler *chain) const {
    llama_sampler_chain_add(chain,
        llama_sampler_init_xtc(_probability, _threshold, _min_keep, resolve_seed(_seed)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────
void godot::register_sampler_types() {
    ClassDB::register_abstract_class<LlamaSamplerBase>();
    ClassDB::register_class<LlamaSamplerGreedy>();
    ClassDB::register_class<LlamaSamplerDist>();
    ClassDB::register_class<LlamaSamplerTemp>();
    ClassDB::register_class<LlamaSamplerTempExt>();
    ClassDB::register_class<LlamaSamplerTopK>();
    ClassDB::register_class<LlamaSamplerTopP>();
    ClassDB::register_class<LlamaSamplerMinP>();
    ClassDB::register_class<LlamaSamplerTypical>();
    ClassDB::register_class<LlamaSamplerPenalties>();
    ClassDB::register_class<LlamaSamplerMirostat>();
    ClassDB::register_class<LlamaSamplerMirostatV2>();
    ClassDB::register_class<LlamaSamplerXTC>();
}
