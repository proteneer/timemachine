{
  inputs = {
    timemachine-flake.url = "github:mcwitt/timemachine-flake";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    { self
    , flake-utils
    , git-hooks
    , nixpkgs
    , timemachine-flake
    , ...
    }: flake-utils.lib.eachSystem
      (with flake-utils.lib.system; [
        x86_64-linux
        x86_64-darwin
      ])
      (system:
      let
        inherit (timemachine-flake.inputs) nixpkgs;
        inherit (nixpkgs) lib;

        pkgs = import nixpkgs {
          inherit system;
          config.allowBroken = true; # needed for jaxlib-bin on darwin
          config.allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [ "cuda_gdb" "cuda_sanitizer_api" ];
          overlays = [ timemachine-flake.overlays.default ];
        };
      in
      {
        devShells.default = timemachine-flake.devShells.${system}.timemachine.override {
          extraPackages = [
            pkgs.basedpyright
            pkgs.py-spy
          ] ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.clang-tools
            pkgs.cudaPackages.cuda_gdb
            pkgs.cudaPackages.cuda_sanitizer_api
            pkgs.gdb
          ];
          extraPythonPackages = ps: with ps; [
            diskcache
            ipywidgets
            jupytext
            memory_profiler
            notebook
            pytest-resource-usage
            pytest-watch
            seaborn
            tqdm
          ];
          extraShellHook = ''
            ${self.checks.${system}.pre-commit-check.shellHook}
          '';
        };

        checks.pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;

          excludes = [
            "\\.pdb"
            "\\.sdf"
            "\\.proto"
            "\\.xml"
            "/vendored/"
            "^attic/"
            "^timemachine/ff/params/"
            "^timemachine/_vendored/"
            "^versioneer\\.py$"
            "^timemachine/_version\\.py$"
            "^timemachine/lib/custom_ops\\.pyi$"
          ];

          hooks = {
            black.enable = true;

            check-yaml = {
              enable = true;
              name = "check yaml";
              description = "checks yaml files for parseable syntax.";
              entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-yaml";
              types = [ "yaml" ];
            };

            clang-format = {
              enable = true;
              files = "^timemachine/cpp/src/";
              types_or = [ "c" "c++" "cuda" ];
            };

            end-of-file-fixer = {
              enable = true;
              name = "fix end of files";
              description = "ensures that a file is either empty, or ends with one newline.";
              entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/end-of-file-fixer";
              types = [ "text" ];
            };

            flake8.enable = true;

            isort.enable = true;

            mypy = {
              enable = true;
              entry = "mypy --ignore-missing-imports --scripts-are-modules";
              types_or = [ "python" "pyi" ];
              require_serial = true;
              excludes = [ "^timemachine/lib/custom_ops.py$" ];
            };

            nixpkgs-fmt.enable = true;

            trailing-whitespace = {
              enable = true;
              name = "trim trailing whitespace";
              description = "trims trailing whitespace.";
              entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer";
              types = [ "text" ];
            };

            verify-typing-stubs = {
              enable = true;
              name = "verify typing stubs";
              entry = "${lib.getExe pkgs.gnumake} build";
              language = "system";
              pass_filenames = false;
              files = "^timemachine/cpp/src/wrap_kernels.cpp$";
            };
          };
        };
      });
}
