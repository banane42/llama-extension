#!/usr/bin/env python
import os, subprocess

env = SConscript("godot-cpp/SConstruct")

# Include dirs
env.Append(CPPPATH=[
    "src/",
    "llama.cpp/include/",
    "llama.cpp/ggml/include/",
])

# Link against the built llama shared lib
env.Append(LIBPATH=[
    "llama.cpp/build/src/Release/",
    "llama.cpp/build/ggml/src/Release/",
])
env.Append(LIBS=["llama", "ggml", "ggml-base", "ggml-cpu"])

sources = Glob("src/*.cpp")
library = env.SharedLibrary(
    "demo/bin/libllama_ext{}{}".format(
        env["suffix"], env["SHLIBSUFFIX"]),
    source=sources,
)
Default(library)