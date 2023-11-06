{
  description = "My flake with dream2nix packages";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-utils,
    fenix,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [fenix.overlays.default];
      };
      #pythonPkgs = pkgs.python311Packages;
      python = pkgs.python311.override {
        packageOverrides = final: prev: {
          huggingface-hub = prev.huggingface-hub.overridePythonAttrs (old: rec {
            version = "0.16.4";
            src = pkgs.fetchFromGitHub {
              owner = "huggingface";
              repo = "huggingface_hub";
              rev = "refs/tags/v${version}";
              hash = "sha256-fWvEvYiaLiVGmDdfibIHJAsu7nUX+eaE0QGolS3LHO8=";
            };
          });

          typer = prev.typer.overridePythonAttrs (old: {
            doCheck = false;
          });

          rich = prev.rich.overridePythonAttrs (old: rec {
            version = "12.4.4";
            src = pkgs.fetchFromGitHub {
              owner = "Textualize";
              repo = "rich";
              rev = "v${version}";
              hash = "sha256-DW6cKJ5bXZdHGzgbYzTS+ryjy71dU9Lcy+egMXL30F8=";
            };
            doCheck = false;
          });

          moto = prev.moto.overridePythonAttrs (old: {
            doCheck = false;
          });

          marshmallow-jsonschema = prev.buildPythonPackage rec {
            pname = "marshmallow-jsonschema";
            version = "0.13.0";
            format = "setuptools";

            #disabled = pythonOlder "3.6";

            src = pkgs.fetchFromGitHub {
              owner = "fuhrysteve";
              repo = pname;
              rev = "138ac8344d4b9dc3a60bb7b90083950c705f3d80";
              hash = "sha256-TgJO47PZu2Xct/pvY/WA9NgvYWFyVUCcDmVCPcYwrB8=";
            };

            propagatedBuildInputs = with prev; [
              marshmallow
              tox
              #setuptools
            ];

            nativeCheckInputs = [
              #pytestCheckHook
            ];

            doCheck = false;

            pythonImportsCheck = [
              "marshmallow_jsonschema"
            ];
          };
          getdaft = let
            rustToolchain = pkgs.fenix.minimal.toolchain;
            rustPlatform = pkgs.makeRustPlatform {
              cargo = rustToolchain;
              rustc = rustToolchain;
            };
          in
            prev.buildPythonPackage rec {
              pname = "getdaft";
              version = "0.2.1";
              format = "pyproject";

              src = pkgs.fetchFromGitHub {
                owner = "Eventual-Inc";
                repo = "Daft";
                rev = "v${version}";
                hash = "sha256-qnsV3OnfwyvrDPelOMlJIkJ7+28yS3ud5zPeoRILcQ0=";
              };

              cargoDeps = rustPlatform.importCargoLock {
                lockFile = "${src}/Cargo.lock";
                outputHashes = {
                  "arrow2-0.17.1" = "sha256-LL7d7m7VGEk0Bophb6VPPqYHi5/KneLwcTaQlfkiCFU="; #pkgs.lib.fakeSha256;
                  "parquet2-0.17.2" = "sha256-o6JuiM4l74bcGgWbnl+YVICjhxJC4BG2WS+eKAvn+mM="; #pkgs.lib.fakeSha256;
                };
              };

              doCheck = false;

              nativeBuildInputs = with pkgs; [
                pkg-config
                perl

                rustToolchain

                rustPlatform.cargoSetupHook
                rustPlatform.maturinBuildHook
              ];

              buildInputs = with pkgs; [
                openssl
              ];

              propagatedBuildInputs = with prev; [
                setuptools
                pyarrow
                fsspec
                psutil
              ];
            };

          torchmetrics = prev.torchmetrics.overridePythonAttrs (old: rec {
            pname = old.pname;
            version = "0.11.4";
            src = prev.fetchPypi {
              inherit pname version;
              hash = "sha256-H+RaFLRN1l2QGZAX3VpLWhKNVqijEdp5FsQCwYxnFJQ";
            };
            doCheck = false;
          });

          torchtext = prev.buildPythonPackage rec {
            pname = "torchtext";
            version = "0.15.2";

            src = pkgs.fetchFromGitHub {
              owner = "pytorch";
              repo = "text";
              rev = "v${version}";
              hash = "sha256-ksik1yYqr20D0QcfSLdDa/A2OMisR64Aif6p8BVsSjc=";
              #hash = pkgs.lib.fakeHash;

              fetchSubmodules = true;
            };

            patchPhase = ''
              substituteInPlace setup.py \
                --replace 'print(" --- Initializing submodules")' "return"
            '';

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              ninja
            ];

            buildInputs = with prev; [
              pybind11
              torch.dev
              torch.lib
            ];

            dontUseCmakeConfigure = true;

            doCheck = false;

            propagatedBuildInputs = with prev; [
              torch
              tqdm
              requests
              numpy
              #torchdata
            ];
          };

          # urllib3 = prev.urllib3.overridePythonAttrs (old: rec {
          #   pname = old.pname;
          #   version = "1.26.16";
          #   format = "setuptools";

          #   src = prev.fetchPypi {
          #     inherit pname version;
          #     hash = "sha256-jxNfZQJ1a95rKpsomJ31++h8mXDOyqaQQe3M5/BYmxQ=";
          #   };
          # });

          torchdata = prev.buildPythonPackage rec {
            pname = "torchdata";
            version = "0.7.0";

            src = pkgs.fetchFromGitHub {
              owner = "pytorch";
              repo = "data";
              rev = "v${version}";
              hash = "sha256-VEakJaehC7BLhhqGqxQYs0lCtfRnYBMjn4DHve3NPFM";

              fetchSubmodules = true;
            };

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              ninja
            ];

            buildInputs = with prev; [
              pybind11
              torch.lib
              torch.dev
            ];

            dontUseCmakeConfigure = true;

            doCheck = false;

            propagatedBuildInputs = with prev; [
              requests
              urllib3
            ];
          };
        };
      };

      ludwig = python.pkgs.buildPythonPackage rec {
        name = "ludwig";
        version = "0.8.6";

        src = pkgs.fetchFromGitHub {
          owner = "ludwig-ai";
          repo = "ludwig";
          rev = "v${version}";
          hash = "sha256-XUFXGGMqYvUuRITi3dHFV/AEql3wsMprq+k3cL1puyY=";
        };

        patchPhase = ''
          substituteInPlace requirements.txt \
            --replace "bitsandbytes<0.41.0" "bitsandbytes" \
            --replace "protobuf==3.20.3" "protobuf" \
            --replace "psutil==5.9.4" "psutil~=5.9.5" \
            --replace "marshmallow-dataclass==8.5.4" "marshmallow-dataclass" \
            --replace "jsonschema>=4.5.0,<4.7" "jsonschema" \
            --replace "PyYAML>=3.12,<6.0.1,!=5.4.*" "PyYAML>=3.12,!=5.4.*"
        '';

        doCheck = false;

        propagatedBuildInputs = with python.pkgs; [
          cython #>=0.25
          h5py #>=2.6,!=3.0.0
          numpy #>=1.15
          pandas #>=1.0,!=1.1.5
          scipy #>=0.18
          tabulate #>=0.7
          scikit-learn
          tqdm
          torch #>=1.13.0
          torchaudio
          torchtext
          torchvision
          torchdata
          pydantic #<2.0
          transformers #>=4.33.2
          tokenizers #>=0.13.3
          spacy #>=2.3
          pyyaml #>=3.12,<6.0.1,!=5.4.* #Exlude PyYAML 5.4.* due to incompatibility with awscli
          absl-py
          kaggle
          requests
          fsspec #[http]<2023.10.0
          dataclasses-json
          jsonschema #>=4.5.0,<4.7
          marshmallow
          marshmallow-jsonschema
          marshmallow-dataclass #==8.5.4
          tensorboard
          nltk # Required for rouge scores.
          torchmetrics #>=0.11.0
          torchinfo
          filelock
          psutil #==5.9.4
          protobuf #==3.20.3
          py-cpuinfo #==9.0.0
          gpustat
          rich #~=12.4.4
          packaging
          retry

          # required for TransfoXLTokenizer when using transformer_xl
          sacremoses
          sentencepiece

          datasets
          huggingface-hub

          commonmark

          bitsandbytes

          getdaft

          xlwt
          xlrd
          pyarrow
          pyxlsb
          openpyxl
          xlsxwriter
          lxml
          html5lib
        ];
      };
      ludwig-app = python.pkgs.toPythonApplication ludwig;
    in rec {
      packages = {
        ludwig = ludwig;
      };
      apps = {
        ludwig-cli = {
          type = "app";
          program = "${ludwig-app}/bin/ludwig";
        };
        default = apps.ludwig-cli;
      };
      defaultPackage = ludwig;
    });
}
