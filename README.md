# Make it stand

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies and create a `makeItStand_bin` binary.

## Run

From within the `build` directory just issue:

    ./makeItStand_bin

A glfw app should launch.

## Dependencies

The only dependencies are stl, eigen, [libigl](http://libigl.github.io/libigl/) and
the dependencies of the `igl::opengl::glfw::Viewer`.

We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.
