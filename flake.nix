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
        config = {
          allowUnfree = true;
          cudaSupport = false; #TODO optional CUDA support
        };
        overlays = [fenix.overlays.default];
      };
      #pythonPkgs = pkgs.python311Packages;
      python = pkgs.python310.override {
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

          google-auth = prev.google-auth.overridePythonAttrs (old: rec {
            pname = old.pname;
            version = "2.21.0";

            src = prev.fetchPypi {
              inherit pname version;
              hash = "sha256-so6ASOV3J+fPDlvY5ydrISrvR2ZUoJURNUqoJ1O0XGY=";
            };

            patches = [
              (pkgs.fetchpatch {
                name = "support-urllib3_2.patch";
                url = "https://github.com/googleapis/google-auth-library-python/commit/9ed006d02d7c9de3e6898ee819648c2fd3367c1d.patch";
                hash = "sha256-64g0GzZeyO8l/s1jqfsogr8pTOBbG9xfp/UeVZNA4q8=";
                includes = ["setup.py" "google/auth/transport/urllib3.py"];
              })
            ];
            postPatch = ''
              substituteInPlace setup.py \
                --replace "urllib3<2.0" "urllib3>=2.0"
            '';
          });

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

              propagatedBuildInputs = with final; [
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

          torchtext = final.buildPythonPackage rec {
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

            buildInputs = with final; [
              pybind11
              torch.dev
              torch.lib
            ];

            dontUseCmakeConfigure = true;

            doCheck = false;

            propagatedBuildInputs = with final; [
              torch
              tqdm
              requests
              numpy
              #torchdata
            ];
          };

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

            buildInputs = with final; [
              pybind11
              torch.lib
              torch.dev
            ];

            dontUseCmakeConfigure = true;

            doCheck = false;

            propagatedBuildInputs = with final; [
              requests
              urllib3
            ];
          };

          loralib = prev.buildPythonPackage rec {
            pname = "loralib";
            version = "0.1.2";
            src = prev.fetchPypi {
              inherit pname version;
              hash = "sha256-Isz/SUpiVLlz3a7p+arUZXlByrQiHHXFoE4MrE+9RWc=";
            };
            propagatedBuildInputs = with final; [
              torch
            ];
          };

          seaborn = prev.seaborn.overridePythonAttrs (old: rec {
            pname = old.pname;
            version = "0.11.0";

            src = prev.fetchPypi {
              inherit pname version;
              sha256 = "390f8437b14f5ce845062f2865ad51656464c306d09bb97d7764c6cba1dd607c";
            };

            nativeBuildInputs = with final; [setuptools];
            checkInputs = with final; [nose];
            propagatedBuildInputs = with final; [pandas matplotlib scipy];

            checkPhase = ''
              nosetests -v
            '';

            # Computationally very demanding tests
            doCheck = false;
          });

          ptitprince = final.buildPythonPackage rec {
            pname = "ptitprince";
            version = "0.2.7";

            src = pkgs.fetchFromGitHub {
              owner = "pog87";
              repo = "PtitPrince";
              rev = "${version}";
              hash = "sha256-EQzNnKC0+Vj7+n8qgy9HwErI0SxgGjBBQ9wnmPp9jKo=";
            };

            propagatedBuildInputs = with final; [
              matplotlib
              numpy #>=1.16
              scipy
              final.seaborn #==0.11
              pandas #>=1.0
              nbconvert #[all]
            ];
          };

          hiplot = final.buildPythonPackage rec {
            pname = "hiplot";
            version = "0.1.33";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-94ob9S+sXcilmsN0VJePc0PnZZrYpBnexMiBU5EmrJ8=";
            };

            propagatedBuildInputs = with final; [
              ipython
              flask
              flask-compress
              beautifulsoup4
            ];
            doCheck = false;
            nativeCheckInputs = with final; [
              mypy
              ipykernel
              wheel
              selenium
              mistune #==0.8.4
              twine
              pkgs.pre-commit
              pandas
              streamlit #>=0.63
              beautifulsoup4
              optuna
              sphinx #==5.2.0
              guzzle_sphinx_theme #==0.7.11
              #m2r2 #==0.3.3
            ];
          };

          fsspec = prev.fsspec.overridePythonAttrs (old: rec {
            version = "2023.4.0";

            src = pkgs.fetchFromGitHub {
              owner = "fsspec";
              repo = "filesystem_spec";
              rev = version;
              hash = "sha256-qkvhmXJNxA8v+kbZ6ulxJAQr7ReQpb+JkbhOUnL59KM=";
            };
          });

          protobuf = prev.protobuf3;

          captum = final.buildPythonPackage rec {
            pname = "captum";
            version = "0.6.0";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-Abs3RiLUa1as1ZBGJ1e/IIftNyHc6/SSLbD6d1SNZSQ=";
            };

            doCheck = false;

            propagatedBuildInputs = with final; [
              matplotlib
              numpy
              torch
            ];

            nativeCheckInputs = with final; [
              pytestCheckHook
              parameterized
            ];
          };

          ray = prev.buildPythonPackage rec {
            pname = "ray";
            version = "2.3.0";
            format = "wheel";

            src = let
              pyVer = "cp310";
              # TODO: parse system value
              os = "manylinux2014";
              arch = "x86_64";
            in
              prev.fetchPypi {
                inherit pname version format;
                dist = pyVer;
                python = pyVer;
                abi = pyVer;
                platform = "${os}_${arch}";
                #hash = "sha256-2GEmtqtRE6O+NSgdVpm61zHI+Q4Cz2VhBXAX35MS8KU="; # cp311
                hash = "sha256-sZ04HUJSWcgLepsUqsnYmGN84mLZG665VpmZ3jsEOWc=";
                #hash = pkgs.lib.fakeHash;
              };

            passthru.optional-dependencies = with final; rec {
              data-deps = [
                pandas
                pyarrow
                fsspec
              ];

              serve-deps = [
                aiorwlock
                fastapi
                pandas
                starlette
                uvicorn
              ];

              tune-deps = [
                tabulate
                tensorboardx
              ];

              rllib-deps =
                tune-deps
                ++ [
                  dm-tree
                  gym
                  lz4
                  matplotlib
                  scikitimage
                  pyyaml
                  scipy
                ];

              air-deps = data-deps ++ serve-deps ++ tune-deps ++ rllib-deps;
            };

            nativeBuildInputs = with final; [
              pkgs.autoPatchelfHook
              pythonRelaxDepsHook
            ];

            pythonRelaxDeps = [
              "click"
              "grpcio"
              "protobuf3"
            ];

            propagatedBuildInputs = with final; [
              attrs
              aiohttp
              aiohttp-cors
              aiosignal
              click
              cloudpickle
              colorama
              colorful
              cython
              filelock
              frozenlist
              gpustat
              grpcio
              jsonschema
              msgpack
              numpy
              opencensus
              packaging
              pkgs.py-spy
              prometheus-client
              protobuf3
              psutil
              pydantic
              pyyaml
              requests
              setproctitle
              smart-open
              virtualenv
            ];

            catchConflicts = false;

            postInstall = ''
              chmod +x $out/${python.sitePackages}/ray/core/src/ray/{gcs/gcs_server,raylet/raylet}
            '';

            pythonImportsCheck = [
              "ray"
            ];
          };

          xgboost-ray = final.buildPythonPackage rec {
            pname = "xgboost-ray";
            version = "0.1.18";

            src = final.fetchPypi {
              inherit version;
              pname = "xgboost_ray";
              hash = "sha256-SV7m1iykrXUvVAegBjcVWdk/OpN6veKAawyXf+0TbKM=";
            };

            propagatedBuildInputs = with final; [
              ray #>=2.0,
              numpy #>=1.16,
              pandas
              wrapt #>=1.12.1,
              xgboost #>=0.90,
              packaging
            ];

            # TODO
            doCheck = false;

            nativeCheckInputs = with final;
              [
                packaging
                # TODO
                #petastorm
                pytest
                pyarrow
                ray #[tune, data, default]
                scikit-learn
                # TODO
                #modin
                dask

                #workaround for now
                protobuf #<4.0.0
                tensorboard #X==2.2
              ]
              ++ final.ray.passthru.optional-dependencies.tune-deps
              ++ final.ray.passthru.optional-dependencies.data-deps;
          };

          modin = final.buildPythonPackage rec {
            pname = "modin";
            version = "0.25.0";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-Ax4d2YKCEvI6h3xwNpAqbhKaepv0P38JUIFe1/M7hZY";
            };

            propagatedBuildInputs = with final; [
              pandas #>=2.1,<2.2
              numpy #>=1.22.4
              # TODO
              #unidist-mpi #>=0.2.1
              mpich
              fsspec #>=2022.05.0
              packaging #>=21.0
              psutil #>=5.8.0
            ];
          };

          lightgbm-ray = final.buildPythonPackage rec {
            pname = "lightgbm-ray";
            version = "0.1.9";

            src = final.fetchPypi {
              inherit version;
              pname = "lightgbm_ray";
              hash = "sha256-rEfxBFXAaTkzWaGOQbZGGFznuadiycQbF78vuqa5NNw=";
            };

            propagatedBuildInputs = with final; [
              lightgbm
              xgboost-ray #>=0.1.12"
              packaging
            ];

            # TODO
            doCheck = false;

            nativeCheckInputs = with final;
              [
                packaging
                parameterized
                # TODO
                #petastorm
                pytest
                pyarrow
                ray #[tune, data, default]
                scikit-learn
                # TODO
                #modin
                xgboost-ray
                #git+https://github.com/ray-project/xgboost_ray.git

                #workaround for now
                protobuf3 #<4.0.0
                tensorboard #X#==2.2
              ]
              ++ final.ray.passthru.optional-dependencies.tune-deps
              ++ final.ray.passthru.optional-dependencies.data-deps;
          };

          hummingbird-ml = final.buildPythonPackage rec {
            pname = "hummingbird-ml";
            version = "0.4.9";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-owN/oV2eG9EaKP0rX/P4h/G6d3VIfTmwJyN9ePNSOxw";
            };

            propagatedBuildInputs = with final; [
              numpy #>=1.15
              onnxconverter-common #>=1.6.0
              scipy
              scikit-learn
              torch #>1.7.0
              psutil
              dill
              protobuf3 #>=3.20.2
            ];

            #preCheck = ''
            #  export HOME=$(mktemp -d)
            #'';

            disabledTestPaths = [
              # fails in sandbox
              "tests/test_sklearn_kneighbors.py"
            ];

            nativeCheckInputs = with final; [
              pytestCheckHook
              flake8
              coverage
              pkgs.pre-commit
            ];

            passthru.optional-dependencies = with final; {
              onnx-deps = [
                onnxruntime #>=1.0.0,
                onnxmltools #>=1.6.0,<=1.11.0,
                skl2onnx #>=1.7.0
              ];
              extra-deps = [
                xgboost #>=0.90<2.0.0
                lightgbm #>=2.2<=3.3.5
                holidays #==0.24
                prophet #==1.1
              ];
            };
          };

          horovod = final.buildPythonPackage rec {
            pname = "horovod";
            version = "0.28.1";

            src = pkgs.fetchFromGitHub {
              owner = pname;
              repo = pname;
              rev = "v${version}";
              hash = "sha256-1Vv5Qen4ChqzHrAbNNAyP4x9YrnzwIYH1N6f5+0nvs4=";
              #hash = pkgs.lib.fakeHash;

              fetchSubmodules = true;
            };

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              ninja
            ];

            dontUseCmakeConfigure = true;

            propagatedBuildInputs = with final; [
              cloudpickle
              psutil
              pyyaml
              packaging
              cffi
            ];

            passthru.optional-dependencies = with final; {
              tensorflow-deps = [
                tensorflow
              ];
              tensorflow-cpu-deps = [
                tensorflowWithoutCuda
              ];
              tensorflow-gpu-deps = [
                tensorflowWithCuda
              ];
              keras-deps = [
                keras
              ];
              pytorch-deps = [
                torch
              ];
              mxnet-deps = [
                mxnet
              ];
              ray-deps = [
                ray
                aioredis
                google-api-core
              ];
              # TODO spark deps
            };

            doCheck = false;

            nativeCheckInputs = with final;
              [
                pytestCheckHook
                mock
                pytest-forked
                pytest-subtests
                parameterized
              ]
              ++ passthru.optional-dependencies.tensorflow-deps
              ++ passthru.optional-dependencies.tensorflow-cpu-deps
              ++ passthru.optional-dependencies.keras-deps
              ++ passthru.optional-dependencies.ray-deps
              ++ passthru.optional-dependencies.mxnet-deps;
          };

          progress-table = final.buildPythonPackage rec {
            pname = "progress-table";
            version = "0.1.27";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-x+i8bD0MAowy9oNLZcIcPGAQGvYrKc8ffvZ8ElA9Zhs=";
            };

            propagatedBuildInputs = with final; [
              colorama
              pandas
            ];

            doCheck = false;
          };

          predibase-api = final.buildPythonPackage rec {
            pname = "predibase-api";
            version = "2023.11.2";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-8yr+tcMjPppEtk/x+A2Y0ym1qyYlNumLcqbiQ+Mcc80=";
            };

            propagatedBuildInputs = with final; [
              protobuf3
            ];
          };

          ipyplot = final.buildPythonPackage rec {
            pname = "ipyplot";
            version = "1.1.1";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-xYB0VHI8gwWB8kqXeEW0b93YlsK3c/stgwx7NL8StBw=";
            };

            propagatedBuildInputs = with final; [
              ipython
              numpy
              pillow
              #bump2version
              shortuuid
              pandas
            ];

            doCheck = false;

            nativeCheckInputs = with final; [
              pytestCheckHook
              pytest-cov
            ];
          };

          predibase = final.buildPythonPackage rec {
            pname = "predibase";
            version = "2023.11.2";
            format = "wheel";

            src = final.fetchPypi {
              inherit pname version;
              hash = "sha256-wnIDLIeUYWqTu6OY7i67JZxxFwbfrAFtmbo+CdBFPuA=";
            };

            nativeBuildInputs = with final; [
              pythonRelaxDepsHook
            ];

            pythonRelaxDeps = [
              "dataclasses-json"
            ];

            propagatedBuildInputs = with final; [
              progress-table
              deprecation
              websockets
              typer
              tritonclient
              tqdm
              tabulate
              pyjwt
              protobuf3
              predibase-api
              ipython
              ipyplot
              pyyaml
              pyarrow
              dataclasses-json
              rich
              urllib3
              requests
              pandas
              python-dateutil
            ];
          };
        };
      };

      ludwig = python.pkgs.buildPythonPackage rec {
        name = "ludwig";
        version = "0.9.0.dev0";

        src = pkgs.fetchFromGitHub {
          owner = "ludwig-ai";
          repo = "ludwig";
          #rev = "v${version}";
          rev = "7458885398dcd2f220ad3f508883fb16cecd6a04";
          hash = "sha256-U4Qtbrl6HPbWGAnQYWRRQDe+20Q4w5DdCHCaRkPgxuQ=";
          #hash = pkgs.lib.fakeHash;
        };

        #--replace "protobuf==3.20.3" "protobuf" \
        patchPhase = ''
          substituteInPlace requirements.txt \
            --replace "bitsandbytes<0.41.0" "bitsandbytes" \
            --replace "psutil==5.9.4" "psutil~=5.9.5" \
            --replace "marshmallow-dataclass==8.5.4" "marshmallow-dataclass" \
            --replace "jsonschema>=4.5.0,<4.7" "jsonschema" \
            --replace "PyYAML>=3.12,<6.0.1,!=5.4.*" "PyYAML>=3.12,!=5.4.*" \

          substituteInPlace requirements_distributed.txt \
            --replace "ray[default,data,serve,tune]>=2.2.0,<2.4" "ray[default,data,serve,tune]>=2.2.0" \
        '';

        doCheck = true;
        catchConflicts = true;

        propagatedBuildInputs = with python.pkgs;
          [
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
            protobuf3 #==3.20.3
            py-cpuinfo #==9.0.0
            gpustat
            rich #~=12.4.4
            packaging
            retry

            #llm

            # required for TransfoXLTokenizer when using transformer_xl
            sacremoses
            sentencepiece

            datasets
            dask
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
          ]
          ++ python.pkgs.fsspec.passthru.optional-dependencies.http;

        pythonImportsCheck = [
          "ludwig"
        ];

        preCheck = ''
          export HOME=$(mktemp -d)
        '';

        nativeCheckInputs = with python.pkgs;
          [
            pytestCheckHook
            pytest
            pytest-timeout
            wget
            six #>=1.13.0
            #aim
            wandb #<0.12.11
            #comet_ml
            mlflow
          ]
          ++ passthru.optional-dependencies.llm
          ++ passthru.optional-dependencies.hyperopt
          ++ passthru.optional-dependencies.viz
          ++ passthru.optional-dependencies.distributed
          ++ passthru.optional-dependencies.serve
          ++ passthru.optional-dependencies.tree
          ++ passthru.optional-dependencies.explain
          ++ passthru.optional-dependencies.extra;

        passthru.optional-dependencies = with python.pkgs; {
          llm = [
            peft
            accelerate
            sentence-transformers
            faiss
            loralib
          ];
          explain = [
            captum
          ];
          extra =
            [
              horovod
              # TODO modin
              predibase
            ]
            ++ horovod.passthru.optional-dependencies.pytorch-deps;
          tree = [
            lightgbm
            lightgbm-ray
            hummingbird-ml #>=0.4.8
          ];
          distributed =
            [
              ray
              dask
            ]
            ++ python.pkgs.dask.passthru.optional-dependencies.dataframe
            ++ python.pkgs.ray.passthru.optional-dependencies.tune-deps
            ++ python.pkgs.ray.passthru.optional-dependencies.serve-deps
            ++ python.pkgs.ray.passthru.optional-dependencies.data-deps;
          hyperopt =
            [
              ray
              hyperopt
            ]
            ++ (with python.pkgs.ray.passthru.optional-dependencies; [(tune-deps ++ serve-deps)]);
          serve = [
            uvicorn
            httpx
            fastapi
            python-multipart
            # TODO
            #neuropod #==0.3.0rc6
          ];
          viz = [
            matplotlib
            seaborn
            hiplot
            ptitprince
          ];
        };
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
      overlays.default = self: prev: {
        ludwig = ludwig-app;
      };
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          (python.withPackages (ps: [ps.pipdeptree]))
          ludwig
        ];
      };
    });
}
