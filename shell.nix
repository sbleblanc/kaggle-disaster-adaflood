let
  nixpkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/f1bad50880bae73ff2d82fafc22010b4fc097a9c.tar.gz") { config = {allowUnfree = true;}; overlays = []; }; #nixos-24.05 2024/08/20";
  nixpkgs-unstable = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/ef7226d68ba45b2de3e428e5d4bb4532caffec7b.tar.gz") { config = {allowUnfree = true;}; overlays = []; }; #nixos-unstable 2024/09/30";
  pkgs = nixpkgs.pkgs;
  nixpkgs-python = import (fetchTarball "https://github.com/cachix/nixpkgs-python/archive/refs/heads/main.zip");
  python = nixpkgs-python.packages.x86_64-linux."3.10.14";
  cuda_pkg = pkgs.cudaPackages.cudatoolkit;
  lib_pkgs = [ pkgs.linuxPackages.nvidia_x11 pkgs.stdenv.cc.cc pkgs.zlib ];
in
pkgs.mkShell {

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath lib_pkgs;

  packages = [
    python
    nixpkgs-unstable.pkgs.uv
    pkgs.cudaPackages.cudatoolkit
    pkgs.zlib
    pkgs.pkg-config
    pkgs.cairo
    pkgs.expat
    pkgs.xorg.libXdmcp
    pkgs.ninja
    pkgs.gobject-introspection
    pkgs.cmake
  ];
}
