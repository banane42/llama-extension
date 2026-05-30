#pragma once
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

// Forward-declare the llama.cpp sampler type so we don't pull in llama.h here.
struct llama_sampler;

namespace godot {

// ─────────────────────────────────────────────────────────────────────────────
// Base class — never instantiated directly, just provides the virtual interface.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerBase : public Resource {
    GDCLASS(LlamaSamplerBase, Resource)
protected:
    static void _bind_methods() {}
public:
    // Subclasses implement this to append themselves onto an existing chain.
    virtual void add_to_chain(llama_sampler *chain) const = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Greedy — always picks the highest-probability token.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerGreedy : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerGreedy, LlamaSamplerBase)
protected:
    static void _bind_methods();
public:
    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Dist — samples from the distribution using a seed.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerDist : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerDist, LlamaSamplerBase)
private:
    int _seed = -1; // -1 maps to LLAMA_DEFAULT_SEED
protected:
    static void _bind_methods();
public:
    void set_seed(int s)    { _seed = s; }
    int  get_seed() const   { return _seed; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Temperature — scales logits before sampling.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerTemp : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerTemp, LlamaSamplerBase)
private:
    float _temperature = 0.8f;
protected:
    static void _bind_methods();
public:
    void  set_temperature(float t)  { _temperature = t; }
    float get_temperature() const   { return _temperature; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Temperature Extended — adds dynamic temperature with exponent and entropy.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerTempExt : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerTempExt, LlamaSamplerBase)
private:
    float _temperature = 0.8f;
    float _delta       = 0.1f;  // dynamic temperature range delta
    float _exponent    = 1.0f;  // dynamic temperature exponent
protected:
    static void _bind_methods();
public:
    void  set_temperature(float t)  { _temperature = t; }
    float get_temperature() const   { return _temperature; }

    void  set_delta(float d)        { _delta = d; }
    float get_delta() const         { return _delta; }

    void  set_exponent(float e)     { _exponent = e; }
    float get_exponent() const      { return _exponent; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Top-K — keeps only the K most likely tokens.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerTopK : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerTopK, LlamaSamplerBase)
private:
    int _top_k = 40;
protected:
    static void _bind_methods();
public:
    void set_top_k(int k)   { _top_k = k; }
    int  get_top_k() const  { return _top_k; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Top-P (nucleus) — keeps tokens whose cumulative probability ≥ p.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerTopP : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerTopP, LlamaSamplerBase)
private:
    float _top_p    = 0.95f;
    int   _min_keep = 1;    // always keep at least this many tokens
protected:
    static void _bind_methods();
public:
    void  set_top_p(float p)        { _top_p = p; }
    float get_top_p() const         { return _top_p; }

    void set_min_keep(int n)        { _min_keep = n; }
    int  get_min_keep() const       { return _min_keep; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Min-P — keeps tokens with probability ≥ p * max_token_prob.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerMinP : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerMinP, LlamaSamplerBase)
private:
    float _min_p    = 0.05f;
    int   _min_keep = 1;
protected:
    static void _bind_methods();
public:
    void  set_min_p(float p)        { _min_p = p; }
    float get_min_p() const         { return _min_p; }

    void set_min_keep(int n)        { _min_keep = n; }
    int  get_min_keep() const       { return _min_keep; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Typical — locally typical sampling.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerTypical : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerTypical, LlamaSamplerBase)
private:
    float _p        = 1.0f;
    int   _min_keep = 1;
protected:
    static void _bind_methods();
public:
    void  set_p(float p)            { _p = p; }
    float get_p() const             { return _p; }

    void set_min_keep(int n)        { _min_keep = n; }
    int  get_min_keep() const       { return _min_keep; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Penalties — penalises recently-seen tokens to reduce repetition.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerPenalties : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerPenalties, LlamaSamplerBase)
private:
    int   _last_n           = 64;
    float _repeat_penalty   = 1.1f;
    float _freq_penalty     = 0.0f;
    float _present_penalty  = 0.0f;
protected:
    static void _bind_methods();
public:
    void  set_last_n(int n)             { _last_n = n; }
    int   get_last_n() const            { return _last_n; }

    void  set_repeat_penalty(float p)   { _repeat_penalty = p; }
    float get_repeat_penalty() const    { return _repeat_penalty; }

    void  set_freq_penalty(float p)     { _freq_penalty = p; }
    float get_freq_penalty() const      { return _freq_penalty; }

    void  set_present_penalty(float p)  { _present_penalty = p; }
    float get_present_penalty() const   { return _present_penalty; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Mirostat v1
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerMirostat : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerMirostat, LlamaSamplerBase)
private:
    int   _n_vocab  = 0;    // 0 = let llama.cpp query the model
    int   _seed     = -1;
    float _tau      = 5.0f;
    float _eta      = 0.1f;
    int   _m        = 100;
protected:
    static void _bind_methods();
public:
    void  set_n_vocab(int n)    { _n_vocab = n; }
    int   get_n_vocab() const   { return _n_vocab; }

    void  set_seed(int s)       { _seed = s; }
    int   get_seed() const      { return _seed; }

    void  set_tau(float t)      { _tau = t; }
    float get_tau() const       { return _tau; }

    void  set_eta(float e)      { _eta = e; }
    float get_eta() const       { return _eta; }

    void  set_m(int m)          { _m = m; }
    int   get_m() const         { return _m; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Mirostat v2
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerMirostatV2 : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerMirostatV2, LlamaSamplerBase)
private:
    int   _seed = -1;
    float _tau  = 5.0f;
    float _eta  = 0.1f;
protected:
    static void _bind_methods();
public:
    void  set_seed(int s)       { _seed = s; }
    int   get_seed() const      { return _seed; }

    void  set_tau(float t)      { _tau = t; }
    float get_tau() const       { return _tau; }

    void  set_eta(float e)      { _eta = e; }
    float get_eta() const       { return _eta; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// XTC — exclude top choices to increase variety / creativity.
// ─────────────────────────────────────────────────────────────────────────────
class LlamaSamplerXTC : public LlamaSamplerBase {
    GDCLASS(LlamaSamplerXTC, LlamaSamplerBase)
private:
    float _probability  = 0.0f;
    float _threshold    = 0.1f;
    int   _min_keep     = 1;
    int   _seed         = -1;
protected:
    static void _bind_methods();
public:
    void  set_probability(float p)  { _probability = p; }
    float get_probability() const   { return _probability; }

    void  set_threshold(float t)    { _threshold = t; }
    float get_threshold() const     { return _threshold; }

    void  set_min_keep(int n)       { _min_keep = n; }
    int   get_min_keep() const      { return _min_keep; }

    void  set_seed(int s)           { _seed = s; }
    int   get_seed() const          { return _seed; }

    void add_to_chain(llama_sampler *chain) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Registration helper — call once from register_types.cpp.
// Registers LlamaSamplerBase and all concrete sampler types with ClassDB.
// ─────────────────────────────────────────────────────────────────────────────
void register_sampler_types();

} // namespace godot
