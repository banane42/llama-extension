#include "register_types.h"
#include "llama_server.h"
#include "llama_samplers.h"
#include "llama_sampler_chain.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

// Singleton instance — created at init, destroyed at uninit
static LlamaServer *_llama_server = nullptr;

void initialize_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;

    // Sampler types and chain — registration contained in their own files
    register_sampler_types();
    register_sampler_chain_type();

    // LLama Server
    ClassDB::register_class<LlamaServer>();
 
    // Instantiate and register as an engine singleton so GDScript can access
    // it via LlamaServer.generate() without any scene setup
    _llama_server = memnew(LlamaServer);
    Engine::get_singleton()->register_singleton("LlamaServer", _llama_server);
}

void uninitialize_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;

    // LLama Server
    Engine::get_singleton()->unregister_singleton("LlamaServer");
    memdelete(_llama_server);
    _llama_server = nullptr;
}

extern "C" {
GDExtensionBool GDE_EXPORT llama_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization)
{
    godot::GDExtensionBinding::InitObject init_obj(
        p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_llama_module);
    init_obj.register_terminator(uninitialize_llama_module);
    init_obj.set_minimum_library_initialization_level(
        MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}
}