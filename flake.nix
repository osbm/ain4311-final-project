{
  description = "Development Shell For AIN4311 Project";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      esp-ppq = pkgs.python3Packages.buildPythonPackage {
        pname = "esp-ppq";
        version = "v0.0.1";

        src = pkgs.fetchFromGitHub {
          owner = "espressif";
          repo = "esp-ppq";
          rev = "1db3d37829bda158909cafec3d9153226f908d59";
          sha256 = "sha256-LPTMS4F2oKbcWe3rejw+lkl9Yopg1vMXTYl6K/mgdQk=";
        };
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python";
        propagatedBuildInputs = with pkgs.python3Packages; [
          numpy
          torchWithCuda
          protobuf
          onnx
          tqdm
        ];
      };
      onnxsim = pkgs.python3Packages.buildPythonPackage {
        pname = "onnxsim";
        version = "0.4.36";
        src = pkgs.fetchFromGitHub {
          owner = "osbm";
          repo = "onnx-simplifier";
          rev = "v0.4.36";
          fetchSubmodules = true;
          sha256 = "sha256-quuTuMlHyMCw7fgDm0MwnFPXq4RJ2zUZJTwNc0chUh4=";
        };
        propagatedBuildInputs = with pkgs.python3Packages; [
          numpy
        ];

        nativeBuildInputs = with pkgs; [
          cmake
          abseil-cpp
          python3Packages.setuptools
        ];
      };
    in
    {
      devShells."${system}".default = pkgs.mkShell {
        packages = with pkgs; [
          (python312.withPackages (ppkgs: with python312Packages; [
            torchWithCuda
            ipython
            numpy
            pandas
            seaborn
            matplotlib
            torchvision
            datasets
            scikit-learn
            jupyter
            notebook
            # esp-ppq
            flatbuffers
            # onnxsim
          ]))
        ];

        shellHook = ''
          echo 'Welcome to the nix development shell.'
          echo "You are using this shell: $SHELL"
          echo "You are using this python: $(which python)"
          export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
        '';
      };
    };
}
