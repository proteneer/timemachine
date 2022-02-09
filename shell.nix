let
  systemPkgs = import <nixpkgs> { };

  sources = import ./nix/sources.nix;

  qchemOverlay = import sources.NixOS-QChem;

  pkgs = import sources.nixpkgs { overlays = [ qchemOverlay ]; };

  cudaPackages = pkgs.cudaPackages_11_6;

  pythonEnv = pkgs.qchem.python3.withPackages (ps:
    let openmm = ps.openmm.override {
      inherit (cudaPackages) cudatoolkit;
      enableCuda = false;
    };
    in
    [
      ps.pip

      openmm
      ps.rdkit

      # install jax with Nix to avoid `ImportError: libstdc++.so.6`
      ps.jax
      ps.jaxlib

      # pip install of grpcio deps hangs; install with Nix instead
      ps.grpcio
      ps.grpcio-tools
    ]);

in
pkgs.mkShell {

  buildInputs = [
    cudaPackages.cudatoolkit
    cudaPackages.cuda_memcheck
    cudaPackages.cuda_sanitizer_api
    pythonEnv

    pkgs.bear
    pkgs.black
    pkgs.clang-tools
    pkgs.cmake
    pkgs.hadolint
    pkgs.pyright
    pkgs.shellcheck
    pkgs.stdenv.cc
  ];

  shellHook = ''
    export CUDACXX=${cudaPackages.cudatoolkit}/bin/nvcc
    export LD_LIBRARY_PATH=${systemPkgs.linuxPackages.nvidia_x11}/lib

    # put packages into $PIP_PREFIX instead of the usual locations.
    # https://nixos.wiki/wiki/Python
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
